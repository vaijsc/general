import torch

# Scores tensor
scores = [1.0000, 0.4813, 0.4610, 0.7018, 0.5019, 0.5404, 0.6593, 0.7835, 0.5713,
    0.5944, 0.4226, 0.4819, 0.4404, 0.6207, 0.4472, 0.4140, 0.4451, 0.4151,
    0.8552, 0.5714, 0.5061, 0.4364, 0.5381, 0.3747, 0.5130, 0.4502, 0.4631,
    0.5230, 0.3710, 0.5629, 0.7407, 0.7220, 0.4343, 0.3787, 0.4537, 0.4529,
    0.5084, 0.5219, 0.5162, 0.4604, 0.4828, 0.3921, 0.4980, 0.3650, 0.5714,
    0.3296, 0.4846, 0.4558, 0.4111, 0.4748, 0.4828, 0.5278, 0.3381, 0.4947,
    0.4615, 0.6622, 0.6165, 0.3952, 0.4397, 0.4579, 0.3383, 0.4846, 0.7366,
    0.3795, 0.4704, 0.4215, 0.3991, 0.1542, 0.4596, 0.3169, 0.4712, 0.4634,
    0.4965, 0.4088, 0.3470, 0.5136, 0.5846, 0.3680, 0.3760, 0.3872, 0.4525,
    0.5010, 0.3401, 0.4424, 0.5130, 0.4180, 0.5590, 0.4485, 0.4665, 0.4472,
    0.3741, 0.3913, 0.4834, 0.8004, 0.4874, 0.4389, 0.4248, 0.4992, 0.3672,
    0.2372, 0.4921, 0.5074, 0.3728, 0.4299, 0.3887, 0.3889, 0.5111, 0.5443,
    0.4084, 0.3815, 0.4794, 0.4020, 0.3561, 0.4156, 0.3187, 0.2734, 0.4073,
    0.4399, 0.4031, 0.4614, 0.4755, 0.4980, 0.3243, 0.4267, 0.3888, 0.4996,
    0.2502, 0.4434, 0.3888, 0.4121, 0.4951, 0.4606, 0.6735, 0.5335, 0.4349,
    0.3664, 0.5008, 0.3166, 0.5191, 0.2922, 0.3875, 0.3412, 0.3755, 0.2789,
    0.4614, 0.4416, 0.3647, 0.4568, 0.4448, 0.4164, 0.5180, 0.3842, 0.2842,
    0.4014, 0.4234, 0.4122, 0.3392, 0.5328, 0.3948, 0.2664, 0.3370, 0.4584,
    0.4509, 0.3704, 0.4286, 0.3813, 0.2967, 0.4016, 0.2449, 0.2468, 0.6114,
    0.2877, 0.2934, 0.4316, 0.2214, 0.2774, 0.4802, 0.4902, 0.3316, 0.3566,
    0.3171, 0.3227, 0.3333, 0.3970, 0.4689, 0.3029, 0.4054, 0.3186, 0.4428,
    0.3983, 0.7371, 0.2965, 0.1938, 0.3239, 0.4141, 0.3716, 0.5082, 0.5648]

# Correcting the list with the full names
INSTANCE_CAT_SCANNET_200 = [
    "chair", "table", "door", "couch", "cabinet", "shelf", "desk", "office chair", "bed", "pillow", 
    "sink", "picture", "window", "toilet", "bookshelf", "monitor", "curtain", "book", "armchair", 
    "coffee table", "box", "refrigerator", "lamp", "kitchen cabinet", "towel", "clothes", "tv", 
    "nightstand", "counter", "dresser", "stool", "cushion", "plant", "ceiling", "bathtub", "end table", 
    "dining table", "keyboard", "bag", "backpack", "toilet paper", "printer", "tv stand", "whiteboard", 
    "blanket", "shower curtain", "trash can", "closet", "stairs", "microwave", "stove", "shoe", 
    "computer tower", "bottle", "bin", "ottoman", "bench", "board", "washing machine", "mirror", 
    "copier", "basket", "sofa chair", "file cabinet", "fan", "laptop", "shower", "paper", "person", 
    "paper towel dispenser", "oven", "blinds", "rack", "plate", "blackboard", "piano", "suitcase", 
    "rail", "radiator", "recycling bin", "container", "wardrobe", "soap dispenser", "telephone", 
    "bucket", "clock", "stand", "light", "laundry basket", "pipe", "clothes dryer", "guitar", 
    "toilet paper holder", "seat", "speaker", "column", "bicycle", "ladder", "bathroom stall", 
    "shower wall", "cup", "jacket", "storage bin", "coffee maker", "dishwasher", "paper towel roll", 
    "machine", "mat", "windowsill", "bar", "toaster", "bulletin board", "ironing board", "fireplace", 
    "soap dish", "kitchen counter", "doorframe", "toilet paper dispenser", "mini fridge", "fire extinguisher", 
    "ball", "hat", "shower curtain rod", "water cooler", "paper cutter", "tray", "shower door", 
    "pillar", "ledge", "toaster oven", "mouse", "toilet seat cover dispenser", "furniture", "cart", 
    "storage container", "scale", "tissue box", "light switch", "crate", "power outlet", "decoration", 
    "sign", "projector", "closet door", "vacuum cleaner", "candle", "plunger", "stuffed animal", 
    "headphones", "dish rack", "broom", "guitar case", "range hood", "dustpan", "hair dryer", 
    "water bottle", "handicap bar", "purse", "vent", "shower floor", "water pitcher", "mailbox", 
    "bowl", "paper bag", "alarm clock", "music stand", "projector screen", "divider", "laundry detergent", 
    "bathroom counter", "object", "bathroom vanity", "closet wall", "laundry hamper", "bathroom stall door", 
    "ceiling light", "trash bin", "dumbbell", "stair rail", "tube", "bathroom cabinet", "cd case", 
    "closet rod", "coffee kettle", "structure", "shower head", "keyboard piano", "case of water bottles", 
    "coat rack", "storage organizer", "folded chair", "fire alarm", "power strip", "calendar", "poster", 
    "potted plant", "luggage", "mattress"
]

for i in range(len(scores)):
    print(f"{INSTANCE_CAT_SCANNET_200[i]}: {scores[i]}")