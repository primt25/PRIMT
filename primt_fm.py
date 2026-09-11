import os
import re
import json
import time
import hashlib
import threading

import numpy as np

LLM_EVAL_SYSTEM = (
    'You are an expert at evaluating the quality of robot motion '
    'trajectories, given structured data at each time step.'
)

VLM_EVAL_SYSTEM = (
    'You are an expert at evaluating the quality of robot trajectories '
    'based on a series of keyframe images, each representing a critical '
    'moment in the trajectory.'
)

_EVAL_STEPS_COMMON = """
Step 2: Preference Label Decision

Based on the previous analysis, compare the two trajectories in terms of goal achievement.

Reply with:
- 0 -> if the goal is better achieved in trajectory 1
- 1 -> if the goal is better achieved in trajectory 2
- -1 -> if both are equal, the difference is unclear, or you are unsure (indecision)

Reply Format
Label: [0 / 1 / -1]

Step 3: Verification and Confidence Score

1. Reflect on the initial analysis and confirm if the chosen label is consistent with the observed actions and outcomes.
2. Assign a confidence score between 0 and 1, where 1 means full confidence and 0 means complete uncertainty.

Reply Format
Final Label: [0 / 1 / -1]
Confidence: [score (0-1)]
Reasoning: [brief explanation of your label and confidence]
"""

LLM_EVAL_USER = """Inputs:

1. Task description: {task_description}

2. Two trajectories, each consisting of {n} steps, formatted as follows:
{format_text}

Trajectory 1:
{text1}

Trajectory 2:
{text2}

Step 1: Trajectory Analysis

For each trajectory:

1. Describe the behaviors.
2. Identify critical events.
3. Evaluate how effectively each trajectory advances toward the goal or indicates failure.

Organize your response clearly by trajectory.
""" + _EVAL_STEPS_COMMON

VLM_EVAL_USER = """Inputs:

1. Task description: {task_description}

2. Images of two trajectories, each represented by a set of keyframes
(provided in order: first all keyframes of trajectory 1, then all
keyframes of trajectory 2).

Step 1: Trajectory Analysis

For each trajectory:

1. Describe the robot behaviors captured in each keyframe.
2. Identify critical events.
3. Assess whether the sequence of actions in each trajectory effectively advances toward the task goal or indicates failure.

Organize your response clearly by trajectory.
""" + _EVAL_STEPS_COMMON

HINDSIGHT_SYSTEM = (
    'You are a reasoning agent specialized in analyzing robot trajectories '
    'and generating counterfactual variants based on structural causal '
    'reasoning.'
)

HINDSIGHT_ABDUCTION_USER = """Inputs:

1. Task description: {task_description}
2. Trajectory Inputs (format):
{format_text}

Trajectory 1:
{text1}

Trajectory 2:
{text2}

3. Preferred Trajectory: {trj_preferred}
4. Keyframe Indices: kvist(1) = {kvist1}, kvist(2) = {kvist2}
5. Preference Reasoning: LLM_reasoning: {llm_reasoning} | VLM_reasoning: {vlm_reasoning}

Step 1: Abduction - Identify Causal Steps

- Analyze the preferred trajectory and determine which time steps are causally responsible for its being preferred.
- Focus on meaningful, task-relevant changes in state-action behavior that likely influenced the preference decision.
- You may use the provided keyframe indices as candidates, but you are not limited to them.

Reply Format
Causal_Steps: [list of identified step indices in the preferred trajectory]
Causal_Reasoning: [brief explanation of why these steps are critical for the preference judgment]
"""

HINDSIGHT_ACTION_USER = """Step 2: Action - Generate Counterfactual Variant

Given:
- Preferred Trajectory
- Causal_Steps and Causal_Reasoning

Your Tasks:

- Select one key step from the identified causal steps.
- Generate a counterfactual trajectory by minimally modifying one state-action feature at that step to reverse the trajectory's preference.
- Light smoothing (2-3 steps before and after) is allowed to maintain continuity.
- All other steps in the trajectory must remain identical to the original.

Reply Format
Edited_Step: [index of the modified step]
Edit_Description: [brief natural language description of the change]
Counterfactual_Trajectory: [text-format trajectory description consistent with the original input format]
"""

FORESIGHT_SYSTEM = (
    'You are a robotics planning agent tasked with generating diverse, '
    'semantically meaningful robot trajectories for simulated manipulation '
    'or locomotion tasks. Your goal is to create high-quality, task-aligned '
    'trajectories that follow physical constraints and can be used for '
    'bootstrapping training.'
)

FORESIGHT_USER = """Inputs:

1. Task description: {task_description}
2. Trajectory Format and Constraints:
{trajectory_format}
{trajectory_constraints}
3. Environment Setup: {environment_setup}

Step 1: High-Level Motion Plan Decomposition

Decompose the task into 3-5 discrete high-level phases. For each phase, clearly specify:

- Phase Name
- Subgoal: What condition or spatial configuration defines success at this phase?
- Motion Strategy: What type of movement or action is needed?
- Constraint Handling: How to account for physical limits, contact, kinematic constraints, etc.

Reply Format
Phase Plan = [
  {{"phase": ..., "subgoal": ..., "strategy": ..., "constraints": ...}},
  ...]

Step 2: Code Translation for Motion Primitives

- Translate the high-level action plan into executable Python code.
- Implement each phase using structured, reusable motion primitives that can adapt to different task contexts.
- Key considerations:
  - Control Logic: Implement continuous control logic for smooth transitions.
  - Parameter Flexibility: Allow inputs of different initial conditions and motion strategy parameters.

Reply Format
{code_return_format}

Step 3: Trajectory Diversification and Sampling

Generate {n} diverse strategy samples by varying both:

- Initial Conditions: Sample from a distribution of start states (e.g., different object locations or robot poses)
- Strategy Parameters: Adjust trajectory-level control parameters (e.g., approach angle, approach height, motion speed)

The code will be run locally once per sample to populate the trajectory buffer with diverse trajectories, so do NOT output the raw trajectory arrays; output only the sample configurations.

Reply Format
Strategy_Samples = [{{
    "traj_id": "fore_001",
    "init_state": {{...}},
    "strategy_config": {{...}}
  }}, ...]
"""


def parse_preference(text):
    if not text:
        return None, 0.0, ''
    label = None
    m = list(re.finditer(r'final\s*label\s*[:\-]?\s*\[?\s*(-1|0|1)',
                         text, re.IGNORECASE))
    if not m:
        m = list(re.finditer(r'\blabel\s*[:\-]?\s*\[?\s*(-1|0|1)',
                             text, re.IGNORECASE))
    if m:
        label = int(m[-1].group(1))
    conf = 0.0
    c = list(re.finditer(r'confidence\s*[:\-]?\s*\[?\s*([01](?:\.\d+)?)',
                         text, re.IGNORECASE))
    if c:
        conf = float(np.clip(float(c[-1].group(1)), 0.0, 1.0))
    reasoning = ''
    r = re.search(r'reasoning\s*[:\-]?\s*(.+)', text,
                  re.IGNORECASE | re.DOTALL)
    if r:
        reasoning = r.group(1).strip()[:500]
    return label, conf, reasoning


def parse_causal_steps(text):
    steps = []
    m = re.search(r'causal_?steps\s*[:=]?\s*\[([^\]]*)\]', text,
                  re.IGNORECASE)
    if m:
        for tok in m.group(1).split(','):
            tok = tok.strip()
            if re.fullmatch(r'\d+', tok):
                steps.append(int(tok))
    reason = ''
    r = re.search(r'causal_?reasoning\s*[:=]?\s*(.+)', text,
                  re.IGNORECASE | re.DOTALL)
    if r:
        reason = r.group(1).strip()[:500]
    return steps, reason


def parse_counterfactual(text):
    step = None
    m = re.search(r'edited_?step\s*[:=]?\s*\[?\s*(\d+)', text, re.IGNORECASE)
    if m:
        step = int(m.group(1))
    desc = ''
    d = re.search(r'edit_?description\s*[:=]?\s*(.+)', text, re.IGNORECASE)
    if d:
        desc = d.group(1).splitlines()[0].strip()[:300]
    traj_text = ''
    t = re.search(r'counterfactual_?trajectory\s*[:=]?\s*(.+)', text,
                  re.IGNORECASE | re.DOTALL)
    if t:
        traj_text = t.group(1)
    return step, desc, traj_text


def parse_code_block(text):
    m = re.findall(r'```(?:python)?\s*(.*?)```', text, re.DOTALL)
    for block in m:
        if 'def generate_trajectory' in block:
            return block
    if m:
        return m[0]
    i = text.find('def ')
    return text[i:] if i >= 0 else None


def parse_strategy_samples(text):
    m = re.search(r'strategy_?samples\s*=?\s*(\[.*)', text,
                  re.IGNORECASE | re.DOTALL)
    blob = m.group(1) if m else text
    depth, end = 0, -1
    for i, ch in enumerate(blob):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return []
    try:
        return json.loads(blob[:end + 1])
    except Exception:
        try:
            fixed = re.sub(r',\s*([\]}])', r'\1',
                           blob[:end + 1].replace("'", '"'))
            return json.loads(fixed)
        except Exception:
            return []


class FMClient(object):
    def __init__(self, backend='openai', llm_model='gpt-4o',
                 vlm_model='gpt-4o', api_key_env='OPENAI_API_KEY',
                 base_url=None, temperature=0.7, max_tokens=2048,
                 max_retries=3, cache_dir='fm_cache',
                 log_path='fm_calls.jsonl', seed=0):
        self.backend = backend
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.log_path = log_path
        self._lock = threading.Lock()
        self._client = None
        self._api_key_env = api_key_env
        self._base_url = base_url
        self.n_calls = 0
        self.n_cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self._mock_rng = np.random.RandomState(seed + 4242)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _lazy_client(self):
        if self._client is None:
            from openai import OpenAI
            for var in ('ALL_PROXY', 'all_proxy'):
                if os.environ.get(var, '').startswith('socks'):
                    os.environ.pop(var)
            key = os.environ.get(self._api_key_env)
            if not key:
                raise RuntimeError(
                    f'FM backend "openai" requires the {self._api_key_env} '
                    'environment variable (paper backbone: gpt-4o). '
                    'Set it, or use primt.fm.backend=mock for offline '
                    'pipeline tests.')
            self._client = OpenAI(api_key=key, base_url=self._base_url)
        return self._client

    def _cache_key(self, model, messages, salt):
        blob = json.dumps({'m': model, 'msgs': messages, 's': salt},
                          sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    @staticmethod
    def _sanitize_messages(messages):
        out = []
        for m in messages:
            c = m.get('content')
            if isinstance(c, list):
                parts = []
                for p in c:
                    if isinstance(p, dict) and p.get('type') == 'image_url':
                        n = len(p.get('image_url', {}).get('url', ''))
                        parts.append({'type': 'image_url',
                                      'image_url': f'<data url, {n} chars>'})
                    elif isinstance(p, dict) and p.get('type') == 'text':
                        parts.append({'type': 'text',
                                      'text': p['text'][:8000]})
                    else:
                        parts.append(p)
                out.append({'role': m.get('role'), 'content': parts})
            else:
                out.append({'role': m.get('role'),
                            'content': str(c)[:8000]})
        return out

    def _log(self, record):
        if not self.log_path:
            return
        record = dict(record)
        record['messages'] = self._sanitize_messages(
            record.get('messages', []))
        record['response'] = str(record.get('response', ''))[:8000]
        with self._lock:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')

    def chat(self, system, user_content, model=None, salt='', kind='',
             max_tokens=None):
        model = model or self.llm_model
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': user_content}]
        return self.chat_messages(messages, model=model, salt=salt,
                                  kind=kind, max_tokens=max_tokens)

    def chat_messages(self, messages, model=None, salt='', kind='',
                      max_tokens=None):
        model = model or self.llm_model
        max_tokens = max_tokens or self.max_tokens
        salt_eff = salt if max_tokens == self.max_tokens \
            else f'{salt}|mt{max_tokens}'
        key = self._cache_key(model, messages, salt_eff)
        cpath = os.path.join(self.cache_dir, key + '.json') \
            if self.cache_dir else None
        if cpath and os.path.exists(cpath):
            with open(cpath) as f:
                self.n_cache_hits += 1
                return json.load(f)['response']

        if self.backend in ('mock', 'scripted'):
            text = self._mock_response(messages)
        else:
            text = self._openai_call(model, messages, max_tokens)
        self.n_calls += 1
        self._log({'ts': time.time(), 'kind': kind, 'model': model,
                   'salt': salt, 'messages': messages, 'response': text})
        if cpath:
            with open(cpath, 'w') as f:
                json.dump({'response': text}, f)
        return text

    def _openai_call(self, model, messages, max_tokens=None):
        client = self._lazy_client()
        err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens)
                usage = getattr(resp, 'usage', None)
                if usage is not None:
                    with self._lock:
                        self.tokens_in += usage.prompt_tokens or 0
                        self.tokens_out += usage.completion_tokens or 0
                return resp.choices[0].message.content
            except Exception as e:
                err = e
                if 'insufficient_quota' in str(e):
                    break
                if 'Connection' in str(e) or 'connect' in str(e).lower():
                    with self._lock:
                        self._client = None
                time.sleep(min(2 ** attempt * 2.0, 30.0))
        raise RuntimeError(f'FM call failed after retries: {err}')

    def _mock_response(self, messages):
        sys_msg = messages[0]['content']
        user = messages[-1]['content']
        user_text = user if isinstance(user, str) else ' '.join(
            p.get('text', '') for p in user if isinstance(p, dict))

        if 'structural causal reasoning' in sys_msg:
            if 'Abduction' in user_text:
                m = re.search(r'kvist\(1\)\s*=\s*\[([^\]]*)\]', user_text)
                cands = [int(x) for x in m.group(1).split(',')] \
                    if m and m.group(1).strip() else [10, 20, 30]
                pick = sorted(self._mock_rng.choice(
                    cands, size=min(3, len(cands)), replace=False).tolist())
                return (f'Causal_Steps: {pick}\n'
                        'Causal_Reasoning: [mock] steps near keyframes '
                        'plausibly drive the preference.')
            prev = ''
            for msg in messages:
                c = msg['content']
                c = c if isinstance(c, str) else ' '.join(
                    p.get('text', '') for p in c if isinstance(p, dict))
                if 'Trajectory 1:' in c:
                    prev = c
            assistant_text = ' '.join(
                str(msg['content']) for msg in messages
                if msg['role'] == 'assistant')
            m = re.search(r'Causal_Steps:\s*\[([\d,\s]*)\]', assistant_text)
            step = int(m.group(1).split(',')[0]) if m and \
                m.group(1).strip() else 10
            pref = re.search(r'Preferred Trajectory:\s*(\d)', prev)
            tag = 'Trajectory %s:' % (pref.group(1) if pref else '1')
            i = prev.find(tag)
            j = prev.find('Trajectory', i + len(tag))
            traj_block = prev[i + len(tag):j if j > 0 else None]

            def bump(mo):
                name, body = mo.group(1), mo.group(2)
                try:
                    vals = [float(x) for x in body.split(',')]
                except ValueError:
                    return mo.group(0)
                for k in range(max(0, step - 2),
                               min(len(vals), step + 3)):
                    if name.startswith('tcp'):
                        vals[k] += 0.06
                    elif name.startswith('action'):
                        vals[k] = 0.0
                return name + ': [' + ', '.join(
                    f'{v:.3f}' for v in vals) + ']'
            edited = re.sub(r'(\w+)\s*:\s*\[([^\]]*)\]', bump, traj_block)
            return (f'Edited_Step: {step}\n'
                    'Edit_Description: [mock] TCP offset + zeroed action '
                    'at the causal step.\n'
                    'Counterfactual_Trajectory:\n' + edited)

        if 'robotics planning agent' in sys_msg:
            return _MOCK_FORESIGHT_RESPONSE

        lab = int(self._mock_rng.choice([0, 1, -1], p=[0.4, 0.4, 0.2]))
        conf = float(self._mock_rng.uniform(0.3, 0.95))
        return ('Step 1: [mock analysis omitted]\n'
                f'Label: {lab}\n'
                f'Final Label: {lab}\n'
                f'Confidence: {conf:.2f}\n'
                'Reasoning: [mock] random pipeline-test response.')


_MOCK_FORESIGHT_RESPONSE = '''Phase Plan = [
  {"phase": "align", "subgoal": "TCP aligned with the button in x-z plane",
   "strategy": "move laterally/vertically to face the button",
   "constraints": "keep clear of the button in y"},
  {"phase": "approach", "subgoal": "TCP close to the button surface",
   "strategy": "advance along +y toward the button",
   "constraints": "smooth deltas, avoid overshoot"},
  {"phase": "press", "subgoal": "button displaced along +y",
   "strategy": "push through the button position",
   "constraints": "sustain contact for several steps"},
  {"phase": "hold", "subgoal": "button stays pressed",
   "strategy": "hold position",
   "constraints": "near-zero velocity"}]

```python
import numpy as np


def linear_segment(start, goal, steps, grip):
    seg = []
    prev = np.asarray(start, dtype=np.float64)
    for i in range(steps):
        a = (i + 1) / float(steps)
        tcp = (1 - a) * np.asarray(start) + a * np.asarray(goal)
        seg.append((tcp.copy(), grip, tcp - prev))
        prev = tcp
    return seg


def generate_trajectory(init_state, strategy_config, env_params):
    """Compose align -> approach -> press -> hold waypoints. Returns an
    (n_steps, 15) array in the required trajectory format."""
    tcp0 = np.asarray(init_state['tcp'], dtype=np.float64)
    btn = np.asarray(init_state['obj_pos'], dtype=np.float64)
    n = int(env_params.get('n_steps', 500))
    ah = float(strategy_config.get('align_height', btn[2]))
    standoff = float(strategy_config.get('standoff', 0.08))
    depth = float(strategy_config.get('press_depth', 0.06))
    speed = float(strategy_config.get('speed', 1.0))

    align = np.array([btn[0], btn[1] - standoff, ah])
    pre = np.array([btn[0], btn[1] - standoff, btn[2]])
    press = np.array([btn[0], btn[1] + depth, btn[2]])

    n_align = max(int(80 / speed), 10)
    n_pre = max(int(40 / speed), 8)
    n_press = max(int(60 / speed), 10)

    wp = []
    wp += linear_segment(tcp0, align, n_align, 1.0)
    wp += linear_segment(align, pre, n_pre, 1.0)
    wp += linear_segment(pre, press, n_press, 1.0)
    while len(wp) < n:
        wp.append((press.copy(), 1.0, np.zeros(3)))
    wp = wp[:n]

    traj = np.zeros((n, 15))
    for t, (tcp, grip, delta) in enumerate(wp):
        traj[t, 0:3] = tcp
        traj[t, 3] = grip
        traj[t, 4:7] = btn
        traj[t, 7:11] = [1, 0, 0, 0]
        traj[t, 11:14] = delta
        traj[t, 14] = 0.0
    return traj
```

Strategy_Samples = [
  {"traj_id": "fore_001", "init_state": {}, "strategy_config":
   {"align_height": 0.12, "standoff": 0.10, "press_depth": 0.06, "speed": 1.0}},
  {"traj_id": "fore_002", "init_state": {}, "strategy_config":
   {"align_height": 0.10, "standoff": 0.06, "press_depth": 0.08, "speed": 0.6}},
  {"traj_id": "fore_003", "init_state": {}, "strategy_config":
   {"align_height": 0.15, "standoff": 0.12, "press_depth": 0.04, "speed": 1.5}},
  {"traj_id": "fore_004", "init_state": {}, "strategy_config":
   {"align_height": 0.12, "standoff": 0.08, "press_depth": 0.00, "speed": 0.8}},
  {"traj_id": "fore_005", "init_state": {}, "strategy_config":
   {"align_height": 0.20, "standoff": 0.15, "press_depth": 0.02, "speed": 2.0}}]
'''
