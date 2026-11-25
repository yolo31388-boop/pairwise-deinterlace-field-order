import pytest
import media.deinterlace_field_order as mod

CASES = [
    ("parse_field_order", (1,)),
    ("interpolate_field_pair", (1,)),
    ("occlusion_fallback", (1,)),
    ("clear_vectors_on_cut", (1,)),
    ("keep_source_rate", (1,)),
    ("review_output", (1,)),
    ("load", (1,)),
    ("detect", (1,)),
    ("deinterlace", (1,)),
    ("export", (1,)),
    ("validate_deint", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
