# Despliegue en AWS App Runner (para después / trabajo)

Mismo código que Cloud Run; cambia la plataforma. Pasos por cada uno de los 3 servicios
(mcp-server, backend, frontend). Requiere cuenta AWS con tarjeta y AWS CLI configurado.

> Costo aproximado: pago por uso (~$0.064/vCPU-hora activa + reposo). 3 servicios
> siempre encendidos ≈ $25–75/mes. Para el trabajo, usar la **cuenta de la empresa**.

## 0. Preparación (una vez)

```bash
aws configure                      # access key, secret, región (ej. us-east-1)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com
```

## 1. Crear repos ECR y subir imágenes

```bash
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
for svc in mcp-server backend frontend; do
  aws ecr create-repository --repository-name validairisk-$svc --region $REGION || true
  docker build -t validairisk-$svc ./$svc
  docker tag validairisk-$svc:latest $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/validairisk-$svc:latest
  docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/validairisk-$svc:latest
done
```

## 2. Crear servicios App Runner

Desde la consola de AWS (App Runner → Create service → Container registry → ECR):

1. **validairisk-mcp**: imagen ECR, puerto 8080, variables `OPENAI_API_KEY`, `ELASTIC_URL`, `ELASTIC_API_KEY`. Copia su URL.
2. **validairisk-backend**: puerto 8000, variables `LLM_PROVIDER`, `OPENAI_API_KEY` (o Bedrock), `ELASTIC_URL/KEY`, `VALIDAIRISK_MCP_URL=<url del mcp>/mcp/`. Copia su URL.
3. **validairisk-frontend**: puerto 8501, variable `BACKEND_URL=<url del backend>`.

## 3. Versión "trabajo" (AWS-nativo)

Cambiar solo variables de entorno (sin tocar código):

- `LLM_PROVIDER=bedrock`, `AWS_REGION`, `BEDROCK_MODEL_ID` (requiere habilitar acceso al modelo en Bedrock).
- RAG: apuntar `ELASTIC_URL` a **Amazon OpenSearch** (compatible con Elasticsearch).
- Datos: los conectores `copiloto_auto`/`validacion_metodologica` ya usan `awswrangler` → **Athena**. El rol IAM del servicio debe permitir Athena + S3 (+ Bedrock).
- Secretos: **AWS Secrets Manager**; memoria de hallazgos: **DynamoDB**.

## Alternativa más simple en AWS

Si App Runner se complica, **ECS no** es necesario: también puedes correr el mismo
contenedor del backend/MCP en **AWS Lambda** (con adaptador) o en una instancia pequeña,
pero App Runner es el camino más directo desde una imagen de contenedor.
