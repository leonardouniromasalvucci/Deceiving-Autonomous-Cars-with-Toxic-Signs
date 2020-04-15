HEIGHT = 32
WIDTH = 32   
IMAGE_SIZE = (HEIGHT, WIDTH)
BATCH_SIZE = 32

N_CHANNEL = 3
INPUT_SHAPE = (1, HEIGHT, WIDTH, N_CHANNEL)
OUTPUT_DIM = 43
EPOCHS = 100
NUM_LABELS = 43

CLASS_SIZE = 2000

ADV_IN_FG_ATTACK = "Adversarial_img/in_distribution/fg/"
ADV_IN_IT_ATTACK = "Adversarial_img/in_distribution/it/"

ADV_OUT_LOGO_FG_ATTACK = "Adversarial_img/out_distribution/logo_signs/fg/target/"
ADV_OUT_LOGO_IT_ATTACK = "Adversarial_img/out_distribution/logo_signs/iterative/target/"

ADV_OUT_BLANKS_FG_ATTACK = "Adversarial_img/out_distribution/blank_signs/fg/target/"
ADV_OUT_BLANKS_IT_ATTACK = "Adversarial_img/out_distribution/blank_signs/iterative/target/"

TRAIN_PATH = "gtsrb-german-traffic-sign/Train"
TRAIN_AUG_PATH = "Train_aug"

MODEL_PATH = "models/mltscl_cnn.hdf5"

DETECTOR_SAMPLES_DIR = "Detector_samples"
CSV_PATH = "sign_name.csv"
CSV_PATH_TEST = "gtsrb-german-traffic-sign/Test.csv"

TEST_DATA_DIR = "gtsrb-german-traffic-sign/Test"
TEST_DATA_DIR_2 = "gtsrb-german-traffic-sign/Testing"
TEST_DATA_DIR_3 = "Samples"

SAMPLE_IMG_DIR = "Original_samples"
SAMPLE_LABEL = 'Original_samples/labels.txt'
SAMPLE_IMG_DIR_BLANK = 'Blank_samples'
SAMPLES_IMG_DIR_LOGO = 'Logo_samples'