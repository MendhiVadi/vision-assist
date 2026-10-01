from vision_assist.config import Config
from vision_assist.scene import Detection, Stabilizer, describe, horizontal_position, _plural
from vision_assist.speaker import Speaker

CFG = Config()
W, H = 640, 480


def det(label, x1, y1, x2, y2):
    return Detection(label, 0.9, (x1, y1, x2, y2))


def test_position():
    assert horizontal_position((0, 0, 100, 100), W) == "on the left"
    assert horizontal_position((270, 0, 370, 100), W) == "in front"
    assert horizontal_position((540, 0, 640, 100), W) == "on the right"


def test_empty():
    assert describe([], W, H, CFG) == "Nothing detected"


def test_plural_and_close():
    d = [det("person", 0, 0, 60, 60), det("person", 10, 10, 70, 70)]
    assert describe(d, W, H, CFG) == "two people on the left"
    assert describe([det("chair", 0, 0, 400, 400)], W, H, CFG).startswith("very close: a chair")


def test_article_an():
    assert describe([det("apple", 300, 200, 340, 240)], W, H, CFG) == "an apple in front"


def test_stabilizer_filters_flicker():
    s = Stabilizer(CFG)
    cup = det("cup", 0, 0, 50, 50)
    assert s.update([cup]) == []
    assert s.update([]) == []
    assert s.update([cup]) == []
    assert s.update([cup]) == [cup]


def test_plurals():
    assert _plural("person", 2) == "people"
    assert _plural("sports ball", 2) == "sports balls"
    assert _plural("bus", 2) == "buses"
    assert _plural("knife", 3) == "knives"
    assert _plural("mouse", 2) == "mice"
    assert _plural("tv", 2) == "TVs"
    assert _plural("scissors", 2) == "scissors"
    assert _plural("skis", 2) == "skis"
    assert _plural("wine glass", 2) == "wine glasses"
    assert _plural("cup", 1) == "cup"


def test_describe_irregular_and_pair():
    d = [det("person", 0, 0, 60, 60), det("person", 10, 10, 70, 70), det("person", 20, 20, 80, 80)]
    assert describe(d, W, H, CFG) == "three people on the left"
    assert describe([det("scissors", 300, 200, 340, 240)], W, H, CFG) == "a pair of scissors in front"


def test_describe_max_four_phrases():
    labels = ["cup", "bottle", "chair", "book", "clock", "vase"]
    d = [det(l, 0, 0, 20, 20) for l in labels]
    assert len(describe(d, W, H, CFG).split(", ")) == 4


def test_stabilizer_keeps_count_when_one_flickers():
    s = Stabilizer(CFG)
    a = det("person", 0, 0, 50, 50)
    b = det("person", 300, 0, 350, 50)
    s.update([a, b])
    s.update([a, b])
    s.update([a, b])
    out = s.update([a])  # b flickers out
    assert len([d for d in out if d.label == "person"]) == 2


def test_stabilizer_suppresses_extra_blip():
    s = Stabilizer(CFG)
    a = det("person", 0, 0, 50, 50)
    b = det("person", 300, 0, 350, 50)
    for _ in range(3):
        s.update([a])
    out = s.update([a, b])  # second person seen only once
    assert out == [a]


def test_stabilizer_drops_label_that_leaves():
    s = Stabilizer(CFG)
    cup = det("cup", 0, 0, 50, 50)
    for _ in range(3):
        s.update([cup])
    for _ in range(3):
        out = s.update([])
    assert out == []


def test_speaker_cooldown():
    sp = Speaker(False, cooldown=3, repeat_after=15)
    assert sp.say("hi", now=10)
    assert not sp.say("other", now=11)
    assert not sp.say("hi", now=14)
    assert sp.say("other", now=14)
    assert sp.say("other", now=30)
