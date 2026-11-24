# Draft email to the INCLUDE authors (for Mukund to send)

**To:** advaithsridhar08@gmail.com. This is the contact the official `ai4bharat/INCLUDE`
Hugging Face card gives ("For any further questions, please reach out to …"). Optionally CC
the AI4Bharat contact listed on their website.

**Subject:** INCLUDE dataset: signer ids for a signer-independent evaluation (+ two small issues)

---

Dear Dr. Sridhar,

I'm a third-year computer science student building a non-commercial, open-source project on
isolated Indian Sign Language word recognition. It runs entirely in the browser, and it is
trained on INCLUDE, which I credit in the README and in the app. Thank you for releasing the
dataset and the official splits.

I'd like to report how well the model generalises to **people it has never seen**, but I
couldn't find signer ids in the video file names, the zip metadata, the split files, or the
Hugging Face card. Would you be able to share a mapping from video path to an **anonymous signer
number** (for example `Animals/1. Dog/MVI_8560.MOV → 3`)? I don't need names or any other
personal information. If signer ids aren't available, a mapping to recording session would also
help. For now I am holding out recording sessions, which I infer from consecutive camera numbers,
and I label that result "session-independent", not "signer-independent".

While working with the data I noticed two things that may be useful to you:

1. The Google Drive link for the train/test split on the Zenodo record
   (`1tGjOR9xbk279ZUYnmk3BE-ktcWBdS24w`) now returns "not found". `download_data.sh` in the
   record also fetches `Train_Test_Split.zip`, which isn't in the record. I'm using the split
   files from the AI4Bharat/INCLUDE GitHub repository instead.
2. The 8 videos of the class **"Second (Number)"** (`Days_and_Time/Second (Number)/MVI_4647–4650`
   and `MVI_5503–5506`) are listed in the split files and the 263-class label map, but they
   aren't in any of the Zenodo zips. If they could be added, the full 263-class benchmark would be
   reproducible from Zenodo alone.

I'd be glad to share the evaluation code and the inventory script that found these, if useful.

Thank you for your time.

Best regards,
Mukund Mangla
[university, programme, year]
[link to the project repository, once public]
