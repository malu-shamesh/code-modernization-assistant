-- Reporting views. setup.sh replaces ${PROJECT} and ${DATASET} before running this file.

CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_cost_by_tier` AS
SELECT
  DATE(timestamp) AS day,
  model,
  tier,
  COUNT(*) AS calls,
  SUM(input_tokens) AS input_tokens,
  SUM(output_tokens) AS output_tokens,
  SUM(input_tokens + output_tokens) AS total_tokens,
  SUM(cost_usd) AS cost_usd
FROM `${PROJECT}.${DATASET}.llm_calls`
GROUP BY day, model, tier;

CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_pass_rate_by_tier` AS
SELECT
  run_id,
  routing_policy,
  routed_tier,
  final_tier,
  COUNT(*) AS modules,
  COUNTIF(success) AS passing_modules,
  SAFE_DIVIDE(COUNTIF(success), COUNT(*)) AS pass_rate
FROM `${PROJECT}.${DATASET}.module_results`
GROUP BY run_id, routing_policy, routed_tier, final_tier;

CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_routing_summary` AS
SELECT
  run_id,
  routing_policy,
  routed_tier,
  COUNT(*) AS modules,
  SAFE_DIVIDE(COUNT(*), SUM(COUNT(*)) OVER (PARTITION BY run_id)) AS share_of_modules,
  COUNTIF(escalated) AS escalated_modules,
  SAFE_DIVIDE(COUNTIF(escalated), COUNT(*)) AS escalation_rate
FROM `${PROJECT}.${DATASET}.module_results`
GROUP BY run_id, routing_policy, routed_tier;

CREATE OR REPLACE VIEW `${PROJECT}.${DATASET}.v_cost_per_passing_module` AS
SELECT
  run_id,
  routing_policy,
  COUNT(*) AS modules,
  COUNTIF(success) AS passing_modules,
  SUM(cost_usd) AS total_cost_usd,
  SAFE_DIVIDE(SUM(cost_usd), COUNT(*)) AS cost_per_module,
  SAFE_DIVIDE(SUM(cost_usd), COUNTIF(success)) AS cost_per_passing_module
FROM `${PROJECT}.${DATASET}.module_results`
GROUP BY run_id, routing_policy;
