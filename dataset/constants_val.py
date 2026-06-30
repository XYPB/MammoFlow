import os
from pathlib import Path


DATA_BASE_DIR = os.environ.get("MAMMOFLOW_DATA_DIR", "data")
DATA_BASE_DIR = os.path.expanduser(DATA_BASE_DIR)
MY_API_TOKEN = os.environ.get("HF_TOKEN")
HF_CKPT_CACHE_DIR = os.getenv("HF_HOME", "~/hf_cache")
HF_CKPT_CACHE_DIR = os.path.expanduser(HF_CKPT_CACHE_DIR)


# #############################################
# MIMIC-CXR-JPG constants
# #############################################
MIMIC_CXR_DATA_DIR = DATA_BASE_DIR + "/mimic-cxr-jpg-resized/2.1.0"
# MIMIC_CXR_TRAIN_TXT = MIMIC_CXR_DATA_DIR / "train.txt"
# MIMIC_CXR_VALID_TXT = MIMIC_CXR_DATA_DIR / "test.txt"
MIMIC_CXR_CHEXPERT_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-chexpert.csv"
MIMIC_CXR_META_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_TEXT_CSV = MIMIC_CXR_DATA_DIR + "/mimic_cxr_sectioned.csv"
MIMIC_CXR_SPLIT_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-split.csv"
# Created csv
MIMIC_CXR_TRAIN_CSV = MIMIC_CXR_DATA_DIR + "/mimic_cxr_lt_labeled_train.csv"
MIMIC_CXR_VALID_CSV = MIMIC_CXR_DATA_DIR + "/mimic_cxr_lt_labeled_val.csv"

# All 3 challenge tasks uses the same train csv
MIMIC_CXR_DATA_C1_DIR = MIMIC_CXR_DATA_DIR + "/starting_k1_test/task1_test_starting_kit"
MIMIC_CXR_TRAIN_CSV_ALL = MIMIC_CXR_DATA_C1_DIR + "/train_labeled.csv"
MIMIC_CXR_TEST_CSV_C1 = MIMIC_CXR_DATA_C1_DIR + "/test_task1.csv"

MIMIC_CXR_DATA_C2_DIR = MIMIC_CXR_DATA_DIR + "/starting_k2_test/task2_test_starting_kit"
MIMIC_CXR_TEST_CSV_C2 = MIMIC_CXR_DATA_C2_DIR + "/test_task2.csv"

MIMIC_CXR_DATA_C3_DIR = MIMIC_CXR_DATA_DIR + "/starting_k3_test/task3_test_starting_kit"
MIMIC_CXR_TEST_CSV_C3 = MIMIC_CXR_DATA_C3_DIR + "/test_task3.csv"

# MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR + "/master.csv"
MIMIC_CXR_MASTER_CSV = MIMIC_CXR_DATA_DIR + "/mimic-cxr-2.0.0-metadata.csv"
MIMIC_CXR_VIEW_COL = "ViewPosition"
MIMIC_CXR_PATH_COL = "fpath"
MIMIC_CXR_SPLIT_COL = "split"
MIMIC_CXR_VIEW_CAPTION = "This is a {{VIEW}} view chest X ray "
MIMIC_CXR_VIEW_CAPTION_TEST = "This is a {{VIEW}} view chest X ray of a patient with "
MIMIC_CXR_LABEL_CAPTION = [
    "This is a chest X ray of a patient with ",
    "This is a chest X ray of a patient without ",
    "There is a high possibility of ",
    "There is no possibility of ",
    "There is a sign of ",
    "There is no sign of ",
    "This is a chest X ray of a patient. The image shows that there is a high possibility of "
]
MIMIC_CXR_LT_FINDINGS = [
    "Adenopathy",
    "Atelectasis",
    "Azygos Lobe",
    "Calcification of the Aorta",
    "Cardiomegaly",
    "Clavicle Fracture",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged Cardiomediastinum",
    "Fibrosis",
    "Fissure",
    "Fracture",
    "Granuloma",
    "Hernia",
    "Hydropneumothorax",
    "Infarction",
    "Infiltration",
    "Kyphosis",
    "Lobar Atelectasis",
    "Lung Lesion",
    "Lung Opacity",
    "Mass",
    "Nodule",
    "Normal",
    "Pleural Effusion",
    "Pleural Other",
    "Pleural Thickening",
    "Pneumomediastinum",
    "Pneumonia",
    "Pneumoperitoneum",
    "Pneumothorax",
    "Pulmonary Embolism",
    "Pulmonary Hypertension",
    "Rib Fracture",
    "Round(ed) Atelectasis",
    "Subcutaneous Emphysema",
    "Support Devices",
    "Tortuous Aorta",
    "Tuberculosis",
]
MIMIC_CXR_LT_FINDINGS_C1 = MIMIC_CXR_LT_FINDINGS
MIMIC_CXR_LT_FINDINGS_C2 = [
    "Atelectasis",
    "Calcification of the Aorta",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Enlarged Cardiomediastinum",
    "Fibrosis",
    "Fracture",
    "Hernia",
    "Infiltration",
    "Lung Lesion",
    "Lung Opacity",
    "Mass",
    "Normal",
    "Nodule",
    "Pleural Effusion",
    "Pleural Other",
    "Pleural Thickening",
    "Pneumomediastinum",
    "Pneumonia",
    "Pneumoperitoneum",
    "Pneumothorax",
    "Subcutaneous Emphysema",
    "Support Devices",
    "Tortuous Aorta",
]
MIMIC_CXR_LT_FINDINGS_C3 = [
    "Bulla",
    "Cardiomyopathy",
    "Hilum",
    "Osteopenia",
    "Scoliosis",
]
MIMIC_CXR_LT_FINDINGS_C3_EXTRA = {
    "Bulla": ["bullae", 
              "lung bulla", 
              "bulla in the lungs / lobes", 
              "bullous emphysema",
              "blebs",],
    "Cardiomyopathy": ["high possibility of cardiomyopathy",
                       "or pericardial effusion"],
    "Hilum": ["hila",
              "hilar mass",
              "hilar enlargement",
              "hilar lymphadenopathy",
              "hilar contours",],
    "Osteopenia": ["osteoporosis",
                   "demineralization",
                   "bony demineralization",
                   "demineralization of the osseous structures",
                   "bones are demineralized",
                   "diffuse osteopenia",
                ],
    "Scoliosis": ["lateral curvature",
                  "curvature of the spine",
                  "curved spine",
                  "scoliosis of the spine",
                  "dextroscoliosis",
                  "levoscoliosis",],
}

MIMIC_CXR_LT_FINDINGS_C3_EXTRA_SENT = {
    "Bulla": [],
    "Cardiomyopathy": ["high possibility of cardiomyopathy",
                       "enlargement of the cardiac silhoutte, suggesting cardiomyopathy",
                       "increased left and right ventricular size, suggesting cardiomyopathy",
                       "enlargement of the cardiac silhouette and increased pulmonary venous congestion, suggesting cardiomyopathy",],
    "Hilum": [],
    "Osteopenia": ["osteoporosis",
                #    "demineralization",
                   "bony demineralization",
                   "demineralization of the osseous structures",
                #    "bones are demineralized",
                   "diffuse osteopenia",
                   "decrease in bone mineral density",
                   "regional osteopenia",
                   "generalized osteopenia"
                   "increased radiolucency of bone"
                ],
    "Scoliosis": [],
}

MIMIC_CXR_LT_FINDINGS_C3_GPT_CAPTION = {
    "Bulla": [
        "A medical image showing a bulla in the lungs.",
        "An x-ray depicting a bulla, a large air-filled space within the lung tissue.",
        "A photo of lung tissue with a visible bulla.",
        "An abnormal, air-filled cavity in the lungs, known as a bulla.",
        "A radiograph highlighting a bulla, which is a large blister-like formation in the lungs.",
        "An image of a chest X-ray where a bulla is present, characterized by a clear, localized area without lung markings.",
    ],
    "Cardiomyopathy": [
        "A heart condition known as cardiomyopathy, where the heart muscle is weakened.",
        "An echocardiogram showing signs of cardiomyopathy.",
        "A medical image depicting the effects of cardiomyopathy on the heart.",
        "A diseased heart muscle, characteristic of cardiomyopathy.",
        "A cardiology scan illustrating the structural changes in the heart due to cardiomyopathy.",
        "An X-ray image displaying signs of cardiomyopathy, such as an abnormally shaped or enlarged heart.",
    ],
    "Hilum": [
        "An anatomical image focusing on the hilum of the lungs.",
        "A x-ray scan showing the hilum, where blood vessels and nerves enter and exit the lungs.",
        "A chest x-ray with a clear view of the pulmonary hilum.",
        "The region of the lung known as the hilum, where major structures converge.",
        "An image highlighting the hilum, the central area of the lung connecting to the heart and other structures.",
        "An image of a chest X-ray focused on the hilum, which appears denser due to the convergence of lung structures.",
    ],
    "Osteopenia": [
        "A bone density x-ray showing signs of osteopenia, where bone density is lower than normal.",
        "An x-ray illustrating the reduced bone mass associated with osteopenia.",
        "A medical image showing early-stage bone thinning, known as osteopenia.",
        "A photo of bones affected by osteopenia, characterized by decreased mineral density.",
        "A diagnosis of osteopenia, depicted by a reduction in bone strength in the image.",
        "An X-ray image displaying signs of osteopenia, with bones appearing less dense than normal.",
    ],
    "Scoliosis": [
        "A spinal x-ray showing the curvature of the spine known as scoliosis.",
        "An image of a spine with scoliosis, where the spine curves to the side.",
        "A radiograph highlighting the abnormal lateral curvature of the spine, indicative of scoliosis.",
        "A medical scan showing scoliosis, a condition where the spine curves abnormally.",
        "A photo of a person's back with a visible curve due to scoliosis.",
        "A chest X-ray illustrates scoliosis, where the spinal column is curved.",
    ],
}

MIMIC_CXR_LT_CAPTION_C3_W_40_1 = "This image likely contains elements of {{TopPredictedClass1}}, {{TopPredictedClass2}}, or {{TopPredictedClass3}}, but it may also include {{UnseenClass}} since this is {{UnseenClassDescription}}."
MIMIC_CXR_LT_CAPTION_C3_W_40_2 = "A medical image showing characteristics of {{TopPredictedClass1}}, {{TopPredictedClass2}}, and {{TopPredictedClass3}}, with signs that could suggest {{UnseenClass}} since this is {{UnseenClassDescription}}."


CLASS_OCCURRENCE = [
    ('Support Devices', 86079.0),
    ('Lung Opacity', 77482.0),
    ('Cardiomegaly', 74738.0),
    ('Pleural Effusion', 66401.0),
    ('Atelectasis', 65376.0),
    ('Pneumonia', 46660.0),
    ('Edema', 37256.0),
    ('Normal', 34292.0),
    ('Enlarged Cardiomediastinum', 29628.0),
    ('Consolidation', 15371.0),
    ('Pneumothorax', 13858.0),
    ('Fracture', 11568.0),
    ('Infiltration', 10087.0),
    ('Rib Fracture', 8919.0),
    ('Nodule', 7531.0),
    ('Mass', 5288.0),
    ('Calcification of the Aorta', 4239.0),
    ('Hernia', 3986.0),
    ('Emphysema', 3661.0),
    ('Adenopathy', 3409.0),
    ('Tortuous Aorta', 3336.0),
    ('Pleural Thickening', 3272.0),
    ('Granuloma', 2965.0),
    ('Fissure', 2803.0),
    ('Lung Lesion', 2338.0),
    ('Tuberculosis', 2078.0),
    ('Subcutaneous Emphysema', 2046.0),
    ('Pulmonary Embolism', 1631.0),
    ('Fibrosis', 1169.0),
    ('Pulmonary Hypertension', 903.0),
    ('Kyphosis', 778.0),
    ('Infarction', 727.0),
    ('Pneumomediastinum', 704.0),
    ('Hydropneumothorax', 646.0),
    ('Pleural Other', 616.0),
    ('Pneumoperitoneum', 516.0),
    ('Azygos Lobe', 199.0),
    ('Round(ed) Atelectasis', 172.0),
    ('Clavicle Fracture', 168.0),
    ('Lobar Atelectasis', 129.0)
]
HEAD_CLS = [c[0] for c in CLASS_OCCURRENCE if c[1] > 0.01 * 257913] # > 1% of the total images
HEAD_POS_CNT = [c[1] for c in CLASS_OCCURRENCE if c[1] > 0.01 * 257913] # > 1% of the total images
HEAD_TOTAL_IMG = sum(HEAD_POS_CNT)
TAIL_CLS = [c[0] for c in CLASS_OCCURRENCE if c[1] <= 0.01 * 257913] # <= 1% of the total images
TAIL_POS_CNT = [c[1] for c in CLASS_OCCURRENCE if c[1] <= 0.01 * 257913] # <= 1% of the total images
TAIL_TOTAL_IMG = sum(TAIL_POS_CNT)


TOTAL_IMG = 257913
CLASS_POS_CNT = [3387, 65099, 198, 4238, 74519, 168, 15276, 37146, 3641, 29580, 1165, 2800, 11537, 2959, 3964, 645, 727, 10081, 770, 129, 2328, 77156, 5263, 7491, 34093, 66134, 615, 3259, 702, 46513, 516, 13825, 1629, 889, 8898, 170, 2045, 85715, 3307, 2078]
CLASS_WEIGHTS = [75.147, 2.9618, 1301.5, 59.8572, 2.46103, 1534.19, 15.8835, 5.9432, 69.835, 7.7191, 220.384, 91.111, 21.3552, 86.162, 64.0, 398.865, 353.763, 24.5840, 333.95, 1998.32, 109.787, 2.3427, 48.004, 33.429, 6.5649, 2.89985, 418.37, 78.138, 366.39, 4.5449, 498.83, 17.6555, 157.325, 289.11, 27.985, 1516.13, 125.118, 2.00895, 76.990, 123.115]


C2_TOTAL_IMG = 193391
C2_CLASS_POS_CNT = [52315, 3420, 59566, 12196, 29849, 2971, 23735, 895, 9360, 3165, 8079, 1878, 61961, 4221, 27412, 6052, 53142, 499, 2668, 579, 37010, 445, 11476, 1731, 69360, 2654]
C2_CLASS_WEIGHT = [2.95727, 59.533, 2.4755, 15.9748, 5.9357, 68.681, 7.72235, 230.312, 21.1180, 64.410, 24.625, 109.236, 2.34121, 48.046, 6.5523, 33.207, 2.895, 413.87, 76.59, 356.55, 4.5937, 464.22, 17.039, 118.598, 1.98478, 77.004,]

# #############################################
# CHEXPERT constants
# #############################################
CHEXPERT_DATA_DIR = DATA_BASE_DIR + "/CheXpert/CheXpert-v1.0"
# CHEXPERT_TRAIN_CSV = CHEXPERT_DATA_DIR + "/train_clean.csv"
CHEXBERT_TRAIN_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_clean_95.csv"
CHEXBERT_TRAIN_5_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_95_five_classes.csv"
CHEXBERT_TRAIN_5_ML_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_95_five_classes_multi_label.csv"
# CHEXBERT_VISUAL_TRAIN_CSV = DATA_BASE_DIR + "/CheXpert/train_visualCheXbert.csv"
CHEXPERT_VALID_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_clean_5.csv"
CHEXPERT_VALID_5_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_5_five_classes.csv"
CHEXPERT_VALID_5_ML_CSV = DATA_BASE_DIR + "/CheXpert/train_cheXbert_5_five_classes_multi_label.csv"
CHEXPERT_TEST_CSV = DATA_BASE_DIR + "/CheXpert/chexpert_5x200.csv"
CHEXPERT_PATH_COL = "Path"
CHEXPERT_VIEW_COL = "AP/PA"
CHEXPERT_FINDINGS = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
    'No Finding',
]
CHEXPERT_FINDINGS_5 = [
    'Atelectasis',
    'Edema',
    'Cardiomegaly',
    'Pleural Effusion',
    'Consolidation',
]
CHEXPERT_BASE_CAPTION = "This is a chest X ray of a patient with "
CHEXPERT_BASE_CAPTION_EMBEDDING = DATA_BASE_DIR + "/CheXpert/chexpert_base_caption_embedding.pt"
CHEXPERT_NEGATIVE_CAPTION = "but this image has no "

# #############################################
# COVIDx-v6 constants
# #############################################
COVIDX_DATA_DIR = DATA_BASE_DIR + "/COVIDx"
COVIDX_TRAIN_DATA_PATH = COVIDX_DATA_DIR + "/train"
COVIDX_TEST_DATA_PATH = COVIDX_DATA_DIR + "/test"
COVIDX_TRAIN_TXT = COVIDX_DATA_DIR + "/train.txt"
COVIDX_TEST_TXT = COVIDX_DATA_DIR + "/test.txt"


# #############################################
# COVID constants
# #############################################
# This is different from the COVIDx dataset
COVID_DATA_DIR = DATA_BASE_DIR + "/COVID/COVID-19_Radiography_Dataset"
COVID_NORMAL_DIR = COVID_DATA_DIR + "/Normal"
COVID_INFECTED_DIR = COVID_DATA_DIR + "/COVID"
COVID_OPACITY_DIR = COVID_DATA_DIR + "/Lung_Opacity"
COVID_PNEUMONIA_DIR = COVID_DATA_DIR + "/Viral Pneumonia"
COVID_CLASSES = [
    "normal chest",
    "COVID-19 infected",
    "lung opacity",
    "viral pneumonia"
]

# #############################################
# RSNA constants
# #############################################
RSNA_DATA_DIR = DATA_BASE_DIR + "/RSNA Pneumonia"
RSNA_TRAIN_DATA_PATH = RSNA_DATA_DIR + "/stage_2_train_images"
RSNA_TEST_DATA_PATH = RSNA_DATA_DIR + "/stage_2_test_images"
RSNA_TRAIN_CSV = RSNA_DATA_DIR + "/stage_2_train_balance.csv"
RSNA_TEST_CSV = RSNA_DATA_DIR + "/stage_2_test_balance.csv"


# #############################################
# EMBED constants
# #############################################
EMBED_DATA_DIR = DATA_BASE_DIR + "/Embed"
EMBED_DATA_PATH = EMBED_DATA_DIR + "/images"
EMBED_TRAIN_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_train.csv"
EMBED_TEST_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_test.csv"
EMBED_VALID_META_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_metadata_reduced_valid.csv"
# Read the full annotation for calcification information
EMBED_ANNO_CSV_REDUCED = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical_reduced.csv"
EMBED_ANNO_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_clinical.csv"
EMBED_LEGENDS_CSV = EMBED_DATA_DIR + "/tables/AWS_Open_Data_Clinical_Legend.csv"
EMBED_LONGITUDINAL_CSV = EMBED_DATA_DIR + "/tables/EMBED_OpenData_longitudinal.csv"
EMBED_INTER_VIEW_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_view.pkl"
EMBED_INTER_SIDE_MAP = EMBED_DATA_DIR + "/tables/img_path2inter_side.pkl"
EMBED_OTHER_VIEW_MAP = EMBED_DATA_DIR + "/tables/img_path2other_view_ccmlo.pkl"
EMBED_BALANCED_TEST_PATH = EMBED_DATA_DIR + "/test_7x200_path2label.pickle"
EMBED_10PCT_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2label.pickle"
EMBED_10PCT_PAIRED_TEST_PATH = EMBED_DATA_DIR + "/test_matched_10pct_path2label.pickle"
EMBED_BALANCED_TRAIN_PATH = EMBED_DATA_DIR + "/train_7x550_path2label.pickle"
EMBED_BALANCED_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x500_path2density.pickle"
EMBED_10PCT_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_10pct_path2density.pickle"
EMBED_10PCT_PAIRED_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_matched_10pct_path2density.pickle"
EMBED_BALANCED_LARGE_DEN_TEST_PATH = EMBED_DATA_DIR + "/test_4x2500_path2density.pickle"
EMBED_BALANCED_DEN_TRAIN_PATH = EMBED_DATA_DIR + "/train_4x1000_path2density.pickle"
EMBED_TRAIN_PATH2DENSITY = EMBED_DATA_DIR + "/train_path2density.pickle"
EMBED_VALID_PATH2DENSITY = EMBED_DATA_DIR + "/valid_path2density.pickle"
EMBED_TEST_PATH2DENSITY = EMBED_DATA_DIR + "/test_path2density.pickle"
EMBED_TRAIN_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_train_resized.pickle"
EMBED_VALID_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_valid_resized.pickle"
EMBED_TEST_ROI_DET_PATH2LABEL = EMBED_DATA_DIR + "/roi2d_path2label_roi_test_resized.pickle"


EMBED_IMAGE_TYPE_COL = "FinalImageType"
EMBED_PATH_COL = "anon_dicom_path"
EMBED_PID_COL = 'empi_anon'
EMBED_SID_COL = 'acc_anon'
EMBED_SIDE_COL = 'ImageLateralityFinal'
EMBED_FINDING_SIDE_COL = 'side'
EMBED_VIEW_COL = 'ViewPosition'
EMBED_DENSITY_COL = 'tissueden'
EMBED_BIRADS_COL = 'asses'
EMBED_PROCEDURE_COL = 'StudyDescription'
EMBED_MASS_SHAPE_COL = 'massshape'
EMBED_MASS_DENSITY_COL = 'massdens'
EMBED_CALC_FIND_COL = 'calcfind'
EMBED_CALC_DIST_COL = 'calcdistri'
EMBED_AGE_COL = 'age_at_study'
EMBED_ROI_COORD = 'ROI_coords'
EMBED_RACE_COL = 'RACE_DESC'
EMBED_ETHNIC_COL = 'ETHNIC_GROUP_DESC'
EMBED_PATH_TRANS_FUNC = lambda x: x.replace("/mnt/NAS2/mammo/anon_dicom", EMBED_DATA_PATH)
EMBED_PROCEDURE2REASON_FUNC = lambda x: "screening" if "screen" in x.lower() else "diagnostic" if "diag" in x.lower() else ""
# Normal caption constants
BREAST_BASE_CAPTION = "This is a breast 2D full-field digital mammogram of a patient "
BREAST_SIDE_CAPTION = "on side " # Make the caption more grammarly correct
BREAST_VIEW_CAPTION = "with view "
BREAST_DENSITY_CAPTION = "with breast tissue density "
BREAST_BIRADS_CAPTION = "with BIRADS score "
# TODO: Add more findings according to the EMBED dataset structure
# Natural Captions
EMBED_NATURE_BASE_CAPTION = "This is a breast 2D full-field digital {{REASON}} mammogram of a patient. "
EMBED_NATURE_IMAGE_CAPTION = "This mammogram is for {{SIDE}} breast with {{VIEW}} view. "
# Structural Captions
EMBED_PROCEDURE = 'Procedure reported: ' # EMBED_PROCEDURE_COL
EMBED_REASON = 'Reason for procedure: ' # Screening / Diagnostic, maybe add more details later
EMBED_PATIENT = 'Patient info: ' # AGE + RACE + ETHNIC
EMBED_IMAGE = 'Image info: ' # EMBED_IMAGE_TYPE_COL + EMBED_SIDE_COL + EMBED_VIEW_COL
EMBED_DENSITY = 'Breast composition: ' # EMBED_DENSITY_COL + extra description
EMBED_FINDINGS = 'Findings: ' # EMBED_MASS info + EMBED_CALC_FIND_COL + extra description
EMBED_IMPRESSIONS = 'Impressions: ' # EMBED_BIRADS_COL + extra description
EMBED_ASSESSMENT = 'Overall Assessment: ' # EMBED_BIRADS_COL number

EMBED_PATIENT_INFO_CAPTION = "This patient is {{RACE}}, {{ETHNIC}}, and {{AGE}} years old. "
EMBED_IMAGE_INFO_CAPTION = "This is a {{IMAGE_TYPE}} full-field digital mammogram of the {{SIDE}} breast with {{VIEW}} view. "
EMBED_BREAST_COMPOSITION_CAPTION = "The breast is {{DENSITY}}. "
EMBED_DENSITY_EXTRA_CAPTION = {
    3: "This may lower the sensitivity of mammography. ",
    4: "This may lower the sensitivity of mammography. ",
}
EMBED_FINDS_CAPTION = "The mammogram shows that "
EMBED_MASS_CAPTION = {
    'A': "an additional imaging is recommended. ",
    'N': "no significant masses, calcification, or other abnormalities are present. ",
    'B': "a benign finding is present. ",
    'P': "a probably benign finding is present. ",
    'S': "a suspicious abnormality is present. ",
    'M': "a highly suggestive of malignancy is present, a biopsy is recommended. ",
    'K': "a known biopsy-proven malignant mass is present. ",
}
EMBED_MASS_EXTRA_CAPTION = 'The mass is {{SHAPE}} and {{DENSITY}}. '
EMBED_CALC_FINDS_CAPTION = 'A {{DISTRI}} {{SHAPE}} calcification is present. '
EMBED_IMPRESSION_CAPTION = "BI-RADS Category {{BIRADS}}: {{BIRADS_DESC}}. "
EMBED_ASSESSMENT_CAPTION = {
    'A': "Additional imaging is recommended. ",
    'N': "Negative. ",
    'B': "Benign. ",
    'P': "Probably benign. ",
    'S': "Suspicious abnormality. ",
    'M': "Highly suggestive of malignancy. ",
    'K': "Known biopsy-proven malignancy. ",
}
EMBED_SIDES_DESC = {
    'L': 'left',
    'R': 'right',
    'B': 'bilateral',
}
EMBED_DENSITY_DESC = {
    1: "almost entirely fat",
    2: "scattered fibroglandular densities",
    3: "heterogeneously dense",
    4: "extremely dense",
    5: "normal male dense",
}
EMBED_LETTER_TO_BIRADS = {
    "A": 0,
    "N": 1,
    "B": 2,
    "P": 3,
    "S": 4,
    "M": 5,
    "K": 6,
}
EMBED_BIRADS_DESC = {
    'A': "additional imaging required",
    'N': "negative",
    'B': "benign finding",
    'P': "probably benign finding",
    'S': "suspicious abnormality",
    'M': "highly suggestive of malignancy",
    'K': "known biopsy-proven malignancy",
}
EMBED_SCREEN_BIRADS_DESC = {
    'A': "additional imaging required",
    'N': "negative",
    'B': "benign finding",
}
GET_JPEG_PATH_FUNC = lambda x: x.replace('Embed', 'EMBED_1080_JPG').replace(".dcm", "_resized.jpg")
GET_ALIGNED_MLO_FUNC = lambda x: x.replace(".jpg", "_align_to_cc.jpg")
EMBED_SCREEN_BIRADS_WEIGHT = [2.1311, 0.4494, 3.2709]
EMBED_SCREEN_DENSITY_WEIGHT = [2.1399, 0.593, 0.6066, 5.0555]
EMBED_ALL_BIRADS_WEIGHT = [1.1503, 0.2315, 0.8455, 2.3951, 5.7693, 62.8517, 45.826]
EMBED_ALL_DENSITY_WEIGHT = [2.2839, 0.6081, 0.5898, 4.5028]


# #############################################
# VinDr constants
# #############################################

VINDR_DATA_PATH = DATA_BASE_DIR + "/vindr-1.0.0"
VINDR_IMAGE_DIR = VINDR_DATA_PATH + "/images"
VINDR_CSV_DIR = VINDR_DATA_PATH + "/breast-level_annotations.csv"
VINDR_DET_CSV_DIR = VINDR_DATA_PATH + "/finding_annotations.csv"
VINDR_DENSITY_LETTER2DIGIT = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4
}
VINDR_BIRADS_DIGIT2LETTER = {
    0: 'N',
    1: 'B',
    2: 'P',
    3: 'S',
    4: 'M'
}
VIN_DR_DENSITY_WEIGHT = [0.9124643059323552, 0.04777300031059451, 0.005967719463259354, 0.03379497429379093]
VIN_DR_BIRADS_WEIGHT = [0.010478014208492025, 0.030028387058222465, 0.15102987146756514, 0.1842069251997844, 0.6242568020659359]
VIN_DR_MASS_WEIGHT = 16.17861187510112
VIN_DR_CALC_WEIGHT = 37.38317757009346

# #############################################
# RSNA constants
# #############################################
RSNA_MAMMO_DATA_PATH = DATA_BASE_DIR + "/rsna-breast-cancer-detection"
RSNA_MAMMO_JPEG_DIR = RSNA_MAMMO_DATA_PATH + "/RSNA_MAMMO_1080_JPG"
RSNA_MAMMO_TRAIN_CSV = RSNA_MAMMO_DATA_PATH + '/rsna_mammo_train.csv'
RSNA_MAMMO_TEST_CSV = RSNA_MAMMO_DATA_PATH + '/rsna_mammo_test.csv'
RSNA_MAMMO_BALANCE_TEST_CSV = RSNA_MAMMO_DATA_PATH + '/rsna_mammo_balanced_test.csv'
RSNA_POS_CLASS_WEIGHT = 46.48148148148148
RSNA_MAMMO_CANCER_DESC = {
    0: "Cancer negative: overall healthy or just benign finding",
    1: "Cancer positive: screening image with known biopsy-proven malignancy or suspicious abnormality found",
}
RSNA_MAMMO_BIRADS_DESC = {
    0: ("N or B", "Negative or Benign"),
    1: ("A", "Additional imaging required with biopsy-proven malignancy or suspicious abnormality found"),
}