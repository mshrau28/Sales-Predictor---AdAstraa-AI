import io
import os
import pandas as pd
import joblib

from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render

from .forms import UploadFileForm
from predictor.ml_utils import add_features



# Path to the saved pipeline (inside predictor_app/model_artifacts)
MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "predictor_app",
    "model_artifacts",
    "final_pipeline.pkl"
)

_model = None

#Load the trained pipeline once and reuse it.
def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def upload_and_predict(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            f = request.FILES["file"]

           
            df = pd.read_csv(f)

            model = get_model()

            preds = model.predict(df)

            df["Predicted_Sale_Amount"] = preds

           
            buffer = io.StringIO()    # Convert DataFrame to CSV in memory
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            response = HttpResponse(   #Return as downloadable file
                buffer.getvalue(),
                content_type="text/csv"
            )
            response["Content-Disposition"] = (
                'attachment; filename="predictions.csv"'
            )
            return response
    else:
        form = UploadFileForm()

    return render(request, "predictor_app/upload.html", {"form": form})
