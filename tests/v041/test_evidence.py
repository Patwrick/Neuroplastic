"""Run guards and exact local provenance, using a disposable isolated Git repo."""
import copy
import hashlib
import json
import subprocess
import zipfile

import pytest

from neuroplastic_v041.experiments import evidence


@pytest.fixture
def config():
    return json.loads((evidence.RELEASE/'configs/cpu_smoke.json').read_text())


def test_shipped_smoke_config_and_schemas(config):
    evidence.validate_m2_config(config)
    manifest=json.loads((evidence.RELEASE/'templates/run_manifest.json').read_text())
    evidence.validate_artifact(manifest,'run_manifest.schema.json')


@pytest.mark.parametrize('section,field,value',[
    ('task','feedback_delays',[1]),('task','contexts_per_lifetime',1),
    ('task','query_feedback',True),('task','target_dim',4),
    ('resources','precision','float64'),('resources','max_job_seconds',181),
    ('resources','max_output_bytes',100000001),('resources','allow_external_downloads',True),
    ('sleep','enabled',False),('sleep','max_steps',5),
    ('training','outer_episodes',0),('training','outer_episodes',3),
    ('training','train_shared_theta',True),('evaluation','max_oracle_calls',9),
    ('graph','columns',128),('learning','admissible_decay',0.5),
    ('learning','beta',float('nan')),
])
def test_unsupported_config_is_rejected(config,section,field,value):
    changed=copy.deepcopy(config)
    changed[section][field]=value
    with pytest.raises((ValueError,evidence.jsonschema.ValidationError)):
        evidence.validate_m2_config(changed)


def initialize_repo(tmp_path,monkeypatch):
    repo=tmp_path/'repository'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.STDOUT)
    git('init')
    (repo/'src').mkdir()
    (repo/'.gitignore').write_text('outputs/\n')
    (repo/'src/old.py').write_text('value = 1\n')
    git('add','.')
    git('-c','user.name=Provenance Test','-c','user.email=test@example.invalid','commit','-m','fixture')
    (repo/'src/old.py').write_text('value = 2\n')
    (repo/'src/new.py').write_text('new_value = 3\n')
    monkeypatch.setattr(evidence,'ROOT',repo)
    return repo


def test_capture_includes_untracked_bytes_and_exact_patch(tmp_path,monkeypatch):
    repo=initialize_repo(tmp_path,monkeypatch)
    out=repo/'outputs/evidence'
    identity=evidence.capture_code(out)
    assert identity['git_dirty'] is True
    patch=(out/'dirty.patch').read_bytes()
    assert b'src/new.py' in patch and b'new_value = 3' in patch
    assert identity['dirty_diff_sha256']==hashlib.sha256(patch).hexdigest()
    manifest=json.loads((out/'source_manifest.json').read_text())
    assert set(manifest)=={'src/old.py','src/new.py'}
    with zipfile.ZipFile(out/'source_snapshot.zip') as archive:
        for name,record in manifest.items():
            value=archive.read(name)
            assert value==(repo/name).read_bytes()
            assert record==dict(sha256=hashlib.sha256(value).hexdigest(),bytes=len(value))
    run_manifest=json.loads((evidence.RELEASE/'templates/run_manifest.json').read_text())
    run_manifest.update(identity)
    evidence.validate_artifact(run_manifest,'run_manifest.schema.json')


def test_capture_rejects_recursive_unignored_output(tmp_path,monkeypatch):
    repo=initialize_repo(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='git-ignored'):
        evidence.capture_code(repo/'src/evidence')


def test_capture_rejects_midcapture_source_change(tmp_path,monkeypatch):
    repo=initialize_repo(tmp_path,monkeypatch)
    real_git=evidence.git
    def changing_git(*args):
        if args[:2]==('diff','HEAD'):
            (repo/'src/old.py').write_text('value = 999\n')
        return real_git(*args)
    monkeypatch.setattr(evidence,'git',changing_git)
    with pytest.raises(RuntimeError,match='Code changed'):
        evidence.capture_code(repo/'outputs/evidence')


def test_optional_peak_memory_is_explicit():
    result=evidence.peak_process_memory()
    assert result['peak_working_set_bytes'] is None or result['peak_working_set_bytes']>0
    assert 'Process-lifetime' in result['scope']
