"""Static architecture audit against the pinned official source; does not train."""
import ast
import subprocess
from pathlib import Path

PINNED_COMMIT = '919ed7e51f9a22604ec2fcefb1525d6fdcfd5bca'


def verify():
    revision = subprocess.check_output(['git', '-C', 'upstream', 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == PINNED_COMMIT, 'Unexpected upstream revision'
    original = Path('upstream/model.py').read_text(encoding='utf-8')
    committed = subprocess.check_output(['git', '-C', 'upstream', 'show', 'HEAD:model.py']).decode('utf-8')
    assert original.replace('\r\n', '\n') == committed.replace('\r\n', '\n'), 'Upstream model modified'
    adapted = Path('model_compat.py').read_text(encoding='utf-8')
    trees = [ast.parse(s) for s in (original, adapted)]
    classes = [{n.name: n for n in tree.body if isinstance(n, ast.ClassDef)} for tree in trees]
    assert classes[0].keys() == classes[1].keys(), 'Model classes changed'
    for name in classes[0]:
        methods = [{n.name: n for n in c[name].body if isinstance(n, ast.FunctionDef)} for c in classes]
        assert methods[0].keys() == methods[1].keys(), f'{name}: methods changed'
        assert ast.dump(methods[0]['__init__']) == ast.dump(methods[1]['__init__']), \
            f'{name}: module/parameter construction changed'
        for method in methods[0]:
            assert ast.dump(methods[0][method].args) == ast.dump(methods[1][method].args), \
                f'{name}.{method}: interface changed'
    print('PASS: official model untouched; all model classes, constructors, parameter definitions,')
    print('layer configuration logic, method names and method interfaces match the pinned source.')
    print('This is a structural audit, not a claim of bitwise-identical training results.')


if __name__ == '__main__':
    verify()
