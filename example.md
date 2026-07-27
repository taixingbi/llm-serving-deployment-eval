# From repo root: load secrets (see .env.example)
set -a
source .env
set +a

#### ecs
SERVICE_URL=$(aws cloudformation describe-stacks \
  --region "${AWS_REGION:-us-east-1}" \
  --stack-name ecs-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" \
  --output text)

curl -sS -N -X POST "${SERVICE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1.0,
    "stream": true
  }'
echo

#### bedrock
FUNCTION_URL=$(aws cloudformation describe-stacks \
  --region "${AWS_REGION:-us-east-1}" \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text)

curl -sS -N -X POST "${FUNCTION_URL}v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 64,
    "temperature": 0,
    "top_p": 1.0,
    "stream": true
  }'
echo

#### self host
curl -sS -N -X POST "${SELFHOST_URL:-http://192.168.86.176:30080}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Reply with one word: ok"}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": true
  }'
echo
