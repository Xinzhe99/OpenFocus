"""Exercise release workflow gates using parsed jobs and isolated job results."""
from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / '.github' / 'workflows'


@pytest.fixture(scope='module')
def bash():
    if sys.platform == 'win32':
        # The Windows system bash may be the WSL launcher, not GitHub's Bash.
        git = shutil.which('git')
        assert git, 'Git is required to locate Git for Windows Bash'
        git_dir = Path(git).resolve().parent
        candidates = [parent / 'bin' / 'bash.exe' for parent in (git_dir.parent, git_dir.parent.parent)]
        executable = next((str(path) for path in candidates if path.is_file()), None)
    else:
        executable = shutil.which('bash')
    assert executable, 'Bash is required to exercise release workflow scripts'
    probe = subprocess.run([executable, '-c', 'exit 0'], capture_output=True)
    assert probe.returncode == 0, (probe.stdout, probe.stderr)
    return executable


class WorkflowLoader(yaml.SafeLoader):
    pass


# GitHub's `on` key is a string, not YAML 1.1's boolean True.
WorkflowLoader.yaml_implicit_resolvers = deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for key, resolvers in WorkflowLoader.yaml_implicit_resolvers.items():
    WorkflowLoader.yaml_implicit_resolvers[key] = [
        item for item in resolvers if item[0] != 'tag:yaml.org,2002:bool']
WorkflowLoader.add_implicit_resolver('tag:yaml.org,2002:bool',
                                    re.compile(r'^(?:true|false)$', re.I), list('tTfF'))


def workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(encoding='utf-8'), Loader=WorkflowLoader)


def expression(value, context):
    """Evaluate the small GitHub expression subset used by these gates."""
    if not isinstance(value, str):
        return value
    value = value.strip()
    if value.startswith('${{') and value.endswith('}}'):
        value = value[3:-2].strip()
    tokens = re.findall(r"\s*(\|\||&&|==|!=|!|\(|\)|'[^']*'|[A-Za-z_][\w.-]*)", value)
    if ''.join(tokens) != re.sub(r'\s+', '', value):
        raise AssertionError(f'unsupported expression: {value}')
    position = 0

    def primary():
        nonlocal position
        token = tokens[position]
        position += 1
        if token == '!':
            return not bool(primary())
        if token == '(':
            result = disjunction()
            assert tokens[position] == ')'
            position += 1
            return result
        if token == 'always':
            assert tokens[position:position + 2] == ['(', ')']
            position += 2
            return True
        if token.startswith("'"):
            return token[1:-1]
        if token in ('true', 'false'):
            return token == 'true'
        result = context
        for part in token.split('.'):
            result = result.get(part, '') if isinstance(result, dict) else ''
        return result

    def comparison():
        nonlocal position
        left = primary()
        if position < len(tokens) and tokens[position] in ('==', '!='):
            operator = tokens[position]
            position += 1
            right = primary()
            return left == right if operator == '==' else left != right
        return left

    def conjunction():
        nonlocal position
        result = comparison()
        while position < len(tokens) and tokens[position] == '&&':
            position += 1
            right = comparison()
            result = right if result else result
        return result

    def disjunction():
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == '||':
            position += 1
            right = conjunction()
            result = result if result else right
        return result

    result = disjunction()
    assert position == len(tokens), value
    return result


def checkout(job):
    return next(step for step in job['steps'] if step.get('uses', '').startswith('actions/checkout@'))


def test_release_calls_suite_and_locks_every_checkout():
    release = workflow('release.yml')
    called = workflow('tests.yml')
    inputs = called['on']['workflow_call']['inputs']
    assert inputs['ref']['type'] == 'string' and inputs['ref']['required'] is True
    assert inputs['run_suite']['type'] == 'boolean' and inputs['run_suite']['default'] is True
    jobs = release['jobs']
    caller = jobs['release-tests']
    assert caller['uses'] == './.github/workflows/tests.yml'
    assert caller['needs'] == 'prepare'
    assert caller['with']['run_suite'] is True
    assert set(jobs['release']['needs']) == {
        'prepare', 'release-tests', 'build-windows', 'build-macos'}
    sha = 'a' * 40
    for channel in ('preview', 'release', 'beta'):
        context = {'needs': {'prepare': {'outputs': {'sha': sha, 'channel': channel}}}}
        assert bool(expression(caller['if'], context)) == (channel in ('release', 'beta'))
        assert bool(expression(jobs['release']['if'], context)) == (channel in ('release', 'beta'))
        assert expression(caller['with']['ref'], context) == sha
        for name in ('build-windows', 'build-macos', 'release'):
            assert expression(checkout(jobs[name])['with']['ref'], context) == sha


def test_packaged_platforms_embed_identity_before_build():
    jobs = workflow('release.yml')['jobs']
    for name in ('build-windows', 'build-macos'):
        steps = jobs[name]['steps']
        marker = next(i for i, step in enumerate(steps)
                      if step.get('name') == 'Write packaged build identity')
        build = next(i for i, step in enumerate(steps)
                     if step.get('name') == 'Build with PyInstaller')
        assert marker < build
        assert '--write-build-identity build_identity.py' in steps[marker]['run']


def test_beta_uses_signing_gate_and_cannot_publish_to_pypi():
    release = workflow('release.yml')['jobs']
    sign = next(step for step in release['build-windows']['steps']
                if step.get('id') == 'signpath')
    pypi = workflow('pypi.yml')
    for channel in ('release', 'beta', 'preview'):
        context = {'needs': {'prepare': {'outputs': {'channel': channel}},
                             'build': {'outputs': {'channel': channel}}}}
        assert bool(expression(sign['if'], context)) == (channel != 'preview')
        assert bool(expression(pypi['jobs']['publish']['if'], context)) == (channel == 'release')
    assert '!v*-beta.*' in pypi['on']['push']['tags']




@pytest.mark.parametrize('channel,failed,expected', [
    ('release', None, True), ('beta', None, True), ('preview', None, False),
    ('release', 'release-tests', False), ('release', 'build-windows', False),
    ('release', 'build-macos', False), ('release', 'prepare', False),
    ('beta', 'release-tests', False), ('beta', 'build-windows', False),
    ('beta', 'build-macos', False),
])
def test_release_job_graph_requires_successful_dependencies(channel, failed, expected):
    job = workflow('release.yml')['jobs']['release']
    results = {name: 'failure' if name == failed else 'success' for name in job['needs']}
    context = {'needs': {'prepare': {'outputs': {'channel': channel}}}}
    # GitHub applies success() implicitly to a job without a status function.
    eligible = all(value == 'success' for value in results.values())
    eligible = eligible and bool(expression(job['if'], context))
    assert eligible is expected


def test_called_suite_conditions_matrix_and_checkout_identity(bash):
    jobs = workflow('tests.yml')['jobs']
    assert set(jobs['test']['strategy']['matrix']['os']) == {
        'ubuntu-latest', 'windows-latest', 'macos-latest'}
    sha = 'b' * 40
    for name in ('test', 'headless-wheel'):
        core = jobs[name]
        for event, enabled, expected in [('push', False, True), ('pull_request', False, True),
                                         ('workflow_dispatch', False, False),
                                         ('workflow_dispatch', True, True)]:
            context = {'github': {'event_name': event, 'sha': 'c' * 40},
                       'inputs': {'ref': sha, 'run_suite': enabled}}
            assert bool(expression(core['if'], context)) == expected
            assert expression(checkout(core)['with']['ref'], context) == sha
        ordinary = {'github': {'sha': 'c' * 40}, 'inputs': {'ref': ''}}
        assert expression(checkout(core)['with']['ref'], ordinary) == 'c' * 40
        verify = next(step for step in core['steps'] if step.get('name') == 'Verify release checkout')
        assert expression(verify['if'], {'inputs': {'ref': sha}})
        assert expression(verify['env']['EXPECTED_SHA'], {'inputs': {'ref': sha}}) == sha
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        for supplied, success in ((actual, True), ('d' * 40, False), ('invalid', False)):
            result = subprocess.run([bash, '-e', '-c', verify['run']], cwd=ROOT,
                                    env={**os.environ, 'EXPECTED_SHA': supplied}, capture_output=True)
            assert (result.returncode == 0) is success, (result.stdout, result.stderr)


@pytest.mark.parametrize('suite,test_result,wheel_result,missing,expected', [
    (True, 'success', 'success', None, True),
    (False, 'success', 'success', None, False),
    (True, 'failure', 'success', None, False),
    (True, 'cancelled', 'success', None, False),
    (True, 'skipped', 'success', None, False),
    (True, 'success', 'failure', None, False),
    (True, 'success', 'success', 'Linux', False),
    (True, 'success', 'success', 'Windows', False),
    (True, 'success', 'success', 'macOS', False),
    (True, 'success', 'success', 'wrong-Linux', False),
    (True, 'success', 'success', 'wrong-Windows', False),
    (True, 'success', 'success', 'wrong-macOS', False),
])
def test_summary_rejects_incomplete_release_suite(tmp_path, bash, suite, test_result, wheel_result, missing, expected):
    summary = workflow('tests.yml')['jobs']['release-summary']
    assert set(summary['needs']) == {'test', 'headless-wheel'}
    assert expression(summary['if'], {'inputs': {'ref': 'a' * 40}})
    assert not expression(summary['if'], {'inputs': {'ref': ''}})
    step = next(item for item in summary['steps'] if item.get('name') == 'Require all release checks')
    results = tmp_path / 'release-test-results'
    results.mkdir()
    for platform in ('Linux', 'Windows', 'macOS'):
        if platform != missing:
            artifact_sha = 'b' * 40 if missing == f'wrong-{platform}' else 'a' * 40
            (results / f'release-test-{platform}').write_text(artifact_sha + '\n', encoding='utf-8')
    context = {'inputs': {'ref': 'a' * 40, 'run_suite': suite},
               'needs': {'test': {'result': test_result},
                         'headless-wheel': {'result': wheel_result}}}
    values = {key: expression(value, context) for key, value in step['env'].items()}
    environment = {**os.environ, **{key: str(value).lower() if isinstance(value, bool)
                                     else str(value) for key, value in values.items()}}
    result = subprocess.run([bash, '-e', '-c', step['run']], cwd=tmp_path,
                            env=environment, capture_output=True)
    assert (result.returncode == 0) is expected, (result.stdout, result.stderr)
