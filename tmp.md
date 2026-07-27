source scripts/resolve_endpoints.sh selfhost
BACKEND=selfhost EXP_ID=smoke_c1_in550_out150 \
  MEAN_IN=550 MEAN_OUT=150 CONC=1 N_REQ=10 \
  ./scripts/run_one.sh


source scripts/resolve_endpoints.sh ecs   # or ecs / bedrock
BACKEND=selfhost EXP_ID=smoke_c1_in550_out150 \
  MEAN_IN=550 MEAN_OUT=150 CONC=1 N_REQ=10 \
  ./scripts/run_one.sh


source scripts/resolve_endpoints.sh bedrock   # or ecs / bedrock
BACKEND=selfhost EXP_ID=smoke_c1_in550_out150 \
  MEAN_IN=550 MEAN_OUT=150 CONC=1 N_REQ=10 \
  ./scripts/run_one.sh