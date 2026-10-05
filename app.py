
import streamlit as st
import joblib
import re
import tempfile
import os

from pypdf import PdfReader

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI Resume Classification System")

st.write(
    "Upload a resume and the machine learning model "
    "will predict its professional category."
)

MODEL_PATH = "resume_classifier.pkl"

model = joblib.load(MODEL_PATH)


def clean_text(text):

    text = str(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9+#./\-\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


if uploaded_file:

    try:

        reader = PdfReader(
            uploaded_file
        )

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

        else:

            st.subheader(
                "Resume Preview"
            )

            st.text_area(
                "Extracted text",
                text[:3000],
                height=250
            )

            cleaned = clean_text(
                text
            )

            prediction = model.predict(
                [cleaned]
            )[0]

            st.success(
                f"Predicted Category: {prediction}"
            )

            # Probability if model supports it
            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    [cleaned]
                )[0]

                classes = model.classes_

                sorted_indices = probabilities.argsort()[::-1][:5]

                st.subheader(
                    "Top Predictions"
                )

                for idx in sorted_indices:

                    st.write(
                        f"**{classes[idx]}** — "
                        f"{probabilities[idx] * 100:.2f}%"
                    )

            else:

                st.info(
                    "The selected model does not provide "
                    "probabilities, so confidence scores "
                    "are not displayed."
                )

    except Exception as e:

        st.error(
            f"Error processing resume: {e}"
        )


st.divider()

st.subheader(
    "About the Model"
)

st.write(
    "The system uses NLP preprocessing and "
    "TF-IDF-based text classification trained "
    "on resume data."
)

st.write(
    "The training pipeline includes data cleaning, "
    "stratified splitting, TF-IDF feature extraction, "
    "model evaluation and error analysis."
)
