"""Build public/models/lens_rules.json from the lens label list (run from the repo root).

The ~4,585-class prompt-free vocabulary is web-tag derived, so people come out as
"jazz artist" / "rocketer" and scenes come out as "autumn" / "disaster". Fixing that
needs labels, not retraining: collapse person-like classes to "person" and drop
non-physical tags.
"""
import json
import re
from pathlib import Path

labels = json.loads(Path("public/models/lens.json").read_text())

ROLE = set("""actor actress adult artist astronomer athlete author baby baker barber bartender
basketballer batter beekeeper biker bodybuilder boxer boy builder businessman businesswoman butcher
camper carpenter catcher chef chemist child climber commander commuter composer cook cowboy cowgirl
craftsman cricketer customer dancer daughter defender dentist designer diver doctor driver drummer
electrician engineer entertainer equestrian explorer farmer father figure fireman firefighter fisherman
florist gardener girl goalkeeper golfer grandfather grandmother grandma grandpa guitarist hacker hiker
historian journalist juggler kid lady lawyer leader magician man manager mechanic miner minister
mother mountaineer musician nurse officer painter paratrooper passenger pedestrian people performer
person philosopher photographer physicist pianist pilot player plumber police policeman politician
preacher producer protester rapper researcher rescuer rider rocker runner saxophonist scientist
shopper singer skater skateboarder skier snowboarder soldier spectator student supporter surfer
swimmer tailor teacher teenager toddler tourist trainer vendor violinist volunteer waiter waitress
warrior welder woman worker wrestler writer youth bassist cellist flutist trumpeter vocalist
instrumentalist percussionist fan guy model supermodel sportsman sportswoman workman worker
bride groom couple crowd family kids children men women boys girls""".split())
# "jazz artist", "rock climber", "ice hockey player" ... but not "tower", "toaster", "computer"
ROLE_PHRASE = re.compile(r"\b(artist|player|actor|actress|performer|singer|musician|driver|worker|photographer|"
                         r"climber|biker|racer|rider|designer|producer|goalkeeper|character|winner|vendor|"
                         r"dancer|skier|skater|surfer|diver|swimmer|athlete|model|girl|boy|woman|man)$")
EXTRA_PERSON = {"rocker", "rapper", "howler", "joker", "hacker", "flasher", "deliver", "batman", "ultraman",
                "strawman", "snowman", "cartoon character", "comic book character", "lead singer",
                "music video performer", "tv actor", "child actor", "theatre actor", "street artist",
                "tattoo artist", "makeup artist", "martial artist", "sport team", "angel", "avatar",
                "fashion girl", "flower girl", "customer", "partner", "leader", "manager", "server"}
NOT_PERSON = {"snowman", "strawman", "batman", "ultraman", "avatar", "angel", "server", "model", "aircraft model",
              "automobile model", "flower girl", "howler", "flasher", "joker", "hacker", "deliver", "partner"}

ABSTRACT = set("""3D-CG-rendering accident act action activity adaptation add adjust adventure advertisement
agriculture aid anniversary appear appearance applause appointment approach argument arrest art
assemble assembly association atmosphere attach attend attraction auction autumn award aid
cancer cheer disaster dinner easter enter leftover letter list makeover number offer order prayer
quarter silver summer spring winter weather thinking thunder twist walking water steam underwater
view waist whisker corner shoulder area angle array border chamber counter container converter
copper entertainment fiber filter lumber meter pointer register shelter thanksgiving""".split())
ABSTRACT_RE = re.compile(r"(film|ceremony|festival|season|show|vector illustration|rendering|concert|"
                         r"competition|tournament|championship|match|race|game|party|wedding|celebration|"
                         r"holiday|christmas|halloween|exhibition|performance|dinner|lunch|breakfast|"
                         r"history|culture|tradition|style|fashion|trend|design|pattern|texture|background|"
                         r"scene|scenery|landscape|view|moment|day|night|time|life|world|business|industry|"
                         r"education|school|class|lesson|training|meeting|event|activity|sport|music|"
                         r"genre|band|team|group|crowd)$")
KEEP = {"night stand", "table", "sports car", "music stand", "school bus", "game controller", "game console",
        "video game", "tablet", "classroom", "window view", "camera view"}

person, drop = [], []
for lab in labels:
    l = lab.lower()
    if lab in KEEP or l in KEEP:
        continue
    last = l.split()[-1]
    if l in EXTRA_PERSON and l not in NOT_PERSON:
        person.append(lab)
    elif l not in NOT_PERSON and (l in ROLE or last in ROLE or ROLE_PHRASE.search(l)):
        person.append(lab)
    elif l in ABSTRACT or ABSTRACT_RE.search(l) or l in NOT_PERSON:
        drop.append(lab)

# stay physical: never drop/convert real objects that merely end in these words
SAFE = {"sheet music", "snowman", "tablet computer", "water bottle", "water cooler", "water heater", "water purifier", "water tower",
        "dish washer", "washer", "toaster", "mixer", "blender", "grinder", "cooler", "heater", "speaker",
        "poker", "pointer", "meter", "counter", "container", "shelter", "filter", "register", "converter",
        "cd player", "mp3 player", "record player", "ceiling fan", "floor fan", "mechanical fan", "scale model",
        "armband", "passport", "terrace", "brace", "waistband", "neckband", "sweatband", "wristband", "elastic band",
        "noseband", "hairstyle", "snowman", "wedding dress", "party hat", "game controller", "music stand", "model car", "model train"}
person = [p for p in person if p.lower() not in SAFE]
drop = [d for d in drop if d.lower() not in SAFE]

MERGE_GROUPS = {
    "hairstyle": ["afro", "bangs", "braid", "curl", "pigtail", "hair", "hair color", "haircut", "hairstyle",
                  "long hair", "short hair", "ponytail", "wig"],
    "ball": ["ball", "baseball", "basketball", "beach ball", "bowling ball", "cricket ball", "rugby ball",
             "handball", "softball", "volleyball", "football", "sports ball"],
    "dog": ["german shepherd", "beagle", "Border collie", "bulldog", "dalmatian", "chihuahua", "cocker", "sheepdog",
            "corgi", "dachshund", "doberman", "French bulldog", "golden retriever", "greyhound", "guard dog",
            "husky", "labrador", "newfoundland", "police dog", "pomeranian", "poodle", "pug", "retriever",
            "rottweiler", "samoyed", "schnauzer", "shepherd", "shiba inu", "street dog", "dog breed"],
    "cat": ["American shorthair", "english shorthair", "persian cat", "siamese", "tabby"],
    "chair": ["armchair", "beach chair", "bean bag chair", "computer chair", "feeding chair", "folding chair",
              "office chair", "rocking chair", "swivel chair"],
    "table": ["changing table", "cocktail table", "side table", "glass table", "kitchen table", "dinning table",
              "picnic table", "round table", "table top", "poker table"],
    "bed": ["bunk bed", "canopy bed", "daybed", "day bed", "infant bed", "hospital bed"],
    "lamp": ["bedside lamp", "table lamp", "oil lamp", "wall lamp", "droplight"],
    "cup": ["coffee cup", "paper cup", "insulated cup", "teacup"],
    "bottle": ["beer bottle", "glass bottle", "thermos bottle", "wine bottle"],
    "bowl": ["glass bowl", "mixing bowl", "salad bowl", "soup bowl", "sugar bowl"],
    "bag": ["clutch bag", "diaper bag", "gift bag", "grocery bag", "messenger bag", "paper bag", "shopping bag",
            "shoulder bag", "tote bag", "handbag"],
    "shoe": ["dress shoe", "leather shoe", "running shoe", "skiing shoes"],
    "boot": ["cowboy boot", "hiking boot", "rain boot", "ski boot"],
    "hat": ["cowboy hat", "dress hat", "straw hat", "sun hat", "plug hat", "christmas hat", "witch hat", "party hat"],
    "cap": ["baseball hat", "golf cap", "jockey cap", "skullcap"],
    "jacket": ["denim jacket", "leather jacket", "ski jacket", "waterproof jacket", "sports coat"],
    "coat": ["trench coat", "fur coat", "lab coat", "overcoat", "raincoat"],
    "shirt": ["dress shirt", "polo shirt", "t shirt", "t-shirt", "sweatshirt"],
    "car": ["concept car", "family car", "muscle car", "race car", "sports car", "model car", "toy car"],
    "bicycle": ["mountain bike", "dirt bike", "stationary bicycle"],
    "phone": ["smartphone", "iphone", "corded phone"],
    "monitor": ["computer monitor", "computer screen", "crt screen", "screen"],
    "computer": ["desktop computer", "computer tower", "computer box"],
    "bird": ["blackbird", "bluebird", "hummingbird", "seabird", "water bird"],
    "fish": ["aquarium fish", "clown fish", "goldfish", "catfish", "flatfish", "blowfish", "swordfish"],
}
merge = {lab: group for group, members in MERGE_GROUPS.items() for lab in members if lab in labels}
drop = [d for d in drop if d not in merge]
person = [p for p in person if p not in merge]

Path("public/models/lens_rules.json").write_text(json.dumps({"person": person, "drop": drop, "merge": merge}))
print(len(person), "person-like;", len(drop), "dropped")
print("PERSON:", " | ".join(person))
print("DROP:", " | ".join(drop))
