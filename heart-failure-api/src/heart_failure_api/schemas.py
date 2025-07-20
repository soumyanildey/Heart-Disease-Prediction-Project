from pydantic import BaseModel

class PatientData(BaseModel):
    Age: int
    Sex: int  # 1 for Male, 0 for Female
    ChestPainType_ASY: int
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    MaxHR: float
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope_Up: int