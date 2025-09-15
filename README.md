# modelo_milp_agilidad

**Deber de Analisis Cuantitativo Para La Gestion Empresarial**

El modelo busca tomar decisiones objetivas y balanceadas sobre qué historias asignar, a quién y en qué secuencia, maximizando valor mientras se cumplen restricciones de capacidad, calidad y cobertura del equipo. Esto permite planificar el sprint de manera eficiente, justa y sostenible.

# Sprint Planning MILP (Python + PuLP)

Este paquete contiene datos sintéticos y un solver MILP para tu caso de planificación de sprint.

## Estructura

- `data/people.csv` — 5 BE, 2 FE, 2 QA, 1 TL, 1 ARQ con 53 h/sprint
- `data/stories.csv` — historias con puntos (1,2,3,5,8), valor y ~3 dependencias
- `data/roles.csv` — distribución de horas por rol (BE 60%, FE 25%, QA 15%), carga de reuniones y horas por bug
- `data/config.yaml` — parámetros (λ, horas por punto, bugs, etc.)
- `solver/solve_sprint.py` — script MILP con PuLP (CBC por defecto)
- `results/` — salidas (CSV + resumen)

## Requisitos

- Python 3.9+
- PuLP (`pip install pulp`) y CBC (instalado con PuLP en la mayoría de entornos)

## Cómo ejecutar

Se recomienda crear un virtual environment para isolar los requerimientos del modelo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -f requirements.txt
python3 solve_sprint.py
```

Resultados en `results/`:

- `selected_stories.csv`
- `assignments.csv`
- `person_utilization.csv`
- `summary.txt`

## Ajustes rápidos

- Cambia `lambda_people_penalty` en `data/config.yaml` para el análisis de sensibilidad.
- Edita `data/stories.csv` para usar tus historias reales de Jira.
- Si deseas cargar kickoff/showme en TL en lugar de QA, ajusta `meeting_load_per_story_hours` en `data/roles.csv`.
