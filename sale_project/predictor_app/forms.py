from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label="Upload test.csv (without Sale_Amount)")
