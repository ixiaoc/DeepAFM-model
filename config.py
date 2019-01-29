# TRAIN_FILE = "data/train_one_train.csv"
# TEST_FILE = "data/train_one_test.csv"
TRAIN_FILE = "data/KDFM_train.csv"
TEST_FILE = "data/KDFM_test.csv"
embedding_file = "data/word_vectors.txt"
embedding_size = 80
train_file = "data/train_two_train.csv"
test_file = "data/train_two_test.csv"
word_file = "data/words.txt"
num_unroll_steps = 100

SUB_DIR = "output"


NUM_SPLITS = 3
RANDOM_SEED = 2018

# types of columns of the dataset dataframe
CATEGORICAL_COLS = []

NUMERIC_COLS = []

IGNORE_COLS = ['rating','name','review_name']

NUM_COLS = ['name','review_name','review_ratting','size_pro','android',
            'genre','rating','full network access','view network connections',
            'prevent device from sleeping','read the contents of your USB storage',
            'modify or delete the contents of your USB storage','receive data from Internet',
            'view Wi-Fi connections','control vibration','run at startup',
            'read phone status and identity','precise location (GPS and network-based)',
            'approximate location (network-based)','take pictures and videos','find accounts on the device',
            'draw over other apps','record audio','read your contacts','connect and disconnect from Wi-Fi',
            'read Google service configuration','pair with Bluetooth devices','retrieve running apps',
            'modify system settings','use accounts on the device','install shortcuts',
            'change your audio settings','access Bluetooth settings','add or remove accounts',
            'change network connectivity','directly call phone numbers','create accounts and set passwords',
            'receive text messages (SMS)','toggle sync on and off','modify your contacts',
            'read your text messages (SMS or MMS)','control flashlight','close other apps',
            'read sync settings','set wallpaper','access USB storage filesystem','disable your screen lock',
            'uninstall shortcuts','send sticky broadcast','read sensitive log data','read call log',
            'read calendar events plus confidential information','control Near Field Communication',
            'send SMS messages','allow Wi-Fi Multicast reception','update component usage statistics',
            'set an alarm','read your own contact card',"add or modify calendar events and send email to guests without owners' knowledge",
            'measure app storage space','download files without notification','read Home settings and shortcuts',
            'read your Web bookmarks and history','reroute outgoing calls','read sync statistics',
            'expand/collapse status bar','delete all app cache data','write call log','read battery statistics',
            'edit your text messages (SMS or MMS)','access extra location provider commands','reorder running apps',
            'adjust your wallpaper size','change system display settings','manage document storage'
]
XM_COLS = ['full network access','view network connections',
            'prevent device from sleeping','read the contents of your USB storage',
            'modify or delete the contents of your USB storage','receive data from Internet',
            'view Wi-Fi connections','control vibration','run at startup',
            'read phone status and identity','precise location (GPS and network-based)',
            'approximate location (network-based)','take pictures and videos','find accounts on the device',
            'draw over other apps','record audio','read your contacts','connect and disconnect from Wi-Fi',
            'read Google service configuration','pair with Bluetooth devices','retrieve running apps',
            'modify system settings','use accounts on the device','install shortcuts',
            'change your audio settings','access Bluetooth settings','add or remove accounts',
            'change network connectivity','directly call phone numbers','create accounts and set passwords',
            'receive text messages (SMS)','toggle sync on and off','modify your contacts',
            'read your text messages (SMS or MMS)','control flashlight','close other apps',
            'read sync settings','set wallpaper','access USB storage filesystem','disable your screen lock',
            'uninstall shortcuts','send sticky broadcast','read sensitive log data','read call log',
            'read calendar events plus confidential information','control Near Field Communication',
            'send SMS messages','allow Wi-Fi Multicast reception','update component usage statistics',
            'set an alarm','read your own contact card',"add or modify calendar events and send email to guests without owners' knowledge",
            'measure app storage space','download files without notification','read Home settings and shortcuts',
            'read your Web bookmarks and history','reroute outgoing calls','read sync statistics',
            'expand/collapse status bar','delete all app cache data','write call log','read battery statistics',
            'edit your text messages (SMS or MMS)','access extra location provider commands','reorder running apps',
            'adjust your wallpaper size','change system display settings','manage document storage'
           ]
TEXT_COLS = ['user_review', 'app_review', 'description']
