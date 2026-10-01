from vision_assist.api import normalize, summarize
from vision_assist.scene import Detection


def det(label, conf=0.9):
    return Detection(label, conf, (0, 0, 10, 10))


def test_aliases_merge_into_everyday_names():
    out = normalize([det("kitchen cabinet"), det("closet"), det("beer can"), det("laptop")])
    assert [d.label for d in out] == ["cupboard", "cupboard", "can", "laptop"]


def test_summarize_counts_and_orders():
    objs = summarize([det("cup", 0.6), det("cup", 0.7), det("watch", 0.9)])
    assert objs[0] == {"label": "cup", "count": 2, "confidence": 0.7}
    assert objs[1]["label"] == "watch"
