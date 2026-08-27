import pandas as pd

from fastapi import (
    FastAPI,
    HTTPException
)

from pydantic import (
    BaseModel,
    Field
)

from typing import List

from caso_berka_model.api.model_loader import (
    load_model,
    get_model_metadata,
    MODEL_NAME
)


# ==========================================
# 1. INICIALIZACIÓN FASTAPI
# ==========================================

app = FastAPI(
    title="API de Clasificación de Clientes - Banco Berka",
    description=(
        "API MLOps para predecir si un cliente "
        "presenta un perfil de buen cliente."
    ),
    version="1.0.0"
)


model = None

model_version_info = {
    "version": "Desconocida",
    "run_id": "Desconocido"
}


# ==========================================
# 2. CARGA DEL MODELO
# ==========================================

@app.on_event("startup")
def startup_event():

    global model
    global model_version_info

    try:

        model = load_model()

        model_version_info = (
            get_model_metadata()
        )

        print(
            f"[FastAPI] Modelo "
            f"{MODEL_NAME} "
            f"v{model_version_info['version']} "
            f"cargado correctamente."
        )

    except Exception as error:

        model = None

        print(
            f"[FastAPI ERROR] "
            f"No se pudo cargar el modelo: "
            f"{error}"
        )


# ==========================================
# 3. ESQUEMA DE ENTRADA
# ==========================================

class ClienteFeatures(BaseModel):

    birth_number: int = Field(
        ...,
        description="Número de nacimiento del cliente"
    )

    date: int = Field(
        ...,
        description="Fecha asociada a la cuenta"
    )

    cantidad_ingresos: float = Field(
        ...,
        ge=0,
        description="Cantidad de operaciones de ingreso"
    )

    total_egresos: float = Field(
        ...,
        ge=0,
        description="Monto total de egresos"
    )

    cantidad_egresos: float = Field(
        ...,
        ge=0,
        description="Cantidad de operaciones de egreso"
    )

    tiene_prestamo: float = Field(
        ...,
        ge=0,
        description="Indica si el cliente posee préstamo"
    )

    monto_prestamo: float = Field(
        ...,
        ge=0,
        description="Monto total del préstamo"
    )

    tiene_tarjeta: float = Field(
        ...,
        ge=0,
        description="Indica si el cliente posee tarjeta"
    )


class PredictionRequest(BaseModel):

    data: List[ClienteFeatures]


# ==========================================
# 4. ENDPOINT PRINCIPAL
# ==========================================

@app.get("/")
def read_root():

    return {
        "status": "Online",
        "project": "Caso Banco Berka",
        "model_name": MODEL_NAME,
        "production_version":
            model_version_info["version"],
        "run_id":
            model_version_info["run_id"]
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy"
            if model is not None
            else "model_not_loaded",

        "model_loaded":
            model is not None
    }


# ==========================================
# 6. PREDICCIÓN
# ==========================================

@app.post("/predict")
def predict(
    payload: PredictionRequest
):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "El modelo no está cargado "
                "en memoria."
            )
        )

    try:

        input_data = pd.DataFrame(
            [
                item.model_dump()
                for item in payload.data
            ]
        )

        # Mantener exactamente
        # las columnas del entrenamiento

        columnas_modelo = [
            "birth_number",
            "date",
            "cantidad_ingresos",
            "total_egresos",
            "cantidad_egresos",
            "tiene_prestamo",
            "monto_prestamo",
            "tiene_tarjeta"
        ]

        input_data = input_data[
            columnas_modelo
        ]

        predictions = model.predict(
            input_data
        )

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_data
                )
            )

        results = []

        for i, pred in enumerate(
            predictions
        ):

            if pred == 1:

                class_label = (
                    "Buen cliente"
                )

            else:

                class_label = (
                    "No buen cliente"
                )

            prob_no_buen_cliente = None
            prob_buen_cliente = None
            confidence = None

            if probabilities is not None:

                prob_no_buen_cliente = (
                    float(
                        probabilities[i][0]
                    )
                )

                prob_buen_cliente = (
                    float(
                        probabilities[i][1]
                    )
                )

                confidence = float(
                    max(
                        probabilities[i]
                    )
                )

            results.append({

                "index":
                    i,

                "prediction_code":
                    int(pred),

                "classification":
                    class_label,

                "confidence_score":
                    round(
                        confidence * 100,
                        2
                    )
                    if confidence is not None
                    else None,

                "probabilities": {

                    "no_buen_cliente":
                        round(
                            prob_no_buen_cliente,
                            4
                        )
                        if prob_no_buen_cliente
                        is not None
                        else None,

                    "buen_cliente":
                        round(
                            prob_buen_cliente,
                            4
                        )
                        if prob_buen_cliente
                        is not None
                        else None
                }
            })

        return {

            "model_metadata": {

                "name":
                    MODEL_NAME,

                "version":
                    model_version_info[
                        "version"
                    ],

                "run_id":
                    model_version_info[
                        "run_id"
                    ]
            },

            "total_predictions":
                len(predictions),

            "results":
                results,

            "message":
                "Inferencia completada correctamente."
        }

    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Error durante la inferencia: "
                f"{error}"
            )
        )