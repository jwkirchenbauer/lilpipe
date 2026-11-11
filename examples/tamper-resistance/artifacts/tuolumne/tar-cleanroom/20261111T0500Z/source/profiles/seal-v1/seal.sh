#!/usr/bin/env bash
# Run only after final reporting and plotting have completed.
set -euo pipefail
[[ $- == *i* ]]
run_path=${1:?absolute run directory required}
[[ "$run_path" == /* ]]
summary_path="$run_path/report/summary.json"
jq -e '
  (.warnings|length)==0 and (.defenses|length)==8 and all(.defenses[];.state=="complete")
  and (.attacks|length)==36 and all(.attacks[];.state=="complete")
  and (.benchmarks|length)==162 and all(.benchmarks[];.state=="complete")
  and (.local_searches|length)==9
  and all(.local_searches[];.state=="complete" and .benchmark_state=="complete" and .strictly_feasible)
  and (.fine_local_searches|length)==1
  and all(.fine_local_searches[];.state=="complete" and .benchmark_state=="complete" and .strictly_feasible and .strict_radius_counterexample)
  and (.peak_wmdp|length)==9 and all(.peak_wmdp[];.validated_attack_points==16)
' "$summary_path" >/dev/null
geometry_files=("$run_path"/geometry/*/geometry.json)
[[ ${#geometry_files[@]} == 9 ]]
jq -se '
  length==9
  and ([.[].planes.reference.reference_definition.identity_sha256]|unique|length)==1
  and ([.[].figure_ce_scale]|unique|length)==1
  and all(.[]; . as $geometry |
    .figure_state=="complete" and (.planes|keys|sort)==(["random","random-step","reference","first","fit"]|sort)
    and all(.planes[];
      (.coordinates|length)==9 and (.grid|length)==81
      and (.point_construction.validation|length)==3
      and all(.point_construction.validation[];
        .parameter_max_abs_difference==0 and .perturbation_relative_l2_difference==0
        and (.ce_absolute_gaps|type)=="object" and all(.ce_absolute_gaps[];.==0))
      and ([.grid[]|select(.x==0 and .y==0)]|length)==1
      and all(.grid[]|select(.x==0 and .y==0);
        ((.forget_train.ce-$geometry.anchor_probes.forget_train.ce)|fabs)<0.00001
        and ((.forget_heldout_subset.ce-$geometry.anchor_probes.forget_heldout_subset.ce)|fabs)<0.00001)
      and all(.grid[];
        .forget_train.ce >= $geometry.figure_ce_scale.ce_min
        and .forget_train.ce <= $geometry.figure_ce_scale.ce_max
        and .forget_heldout_subset.ce >= $geometry.figure_ce_scale.ce_min
        and .forget_heldout_subset.ce <= $geometry.figure_ce_scale.ce_max)))
' "${geometry_files[@]}" >/dev/null
rg -q '^Ran 72 tests' "$run_path/final-tests.log"
rg -q '^OK \(skipped=2\)' "$run_path/final-tests.log"
cp "$run_path"/launches/geometry/cr-geometry-cache-test/logs/*.out "$run_path/gpu-cache-test.log"
cp "$run_path"/launches/validation/cr-final-multi-device-unit/logs/*.out "$run_path/gpu-gradient-test.log"
rg -q 'OK' "$run_path/gpu-cache-test.log"
rg -q 'OK' "$run_path/gpu-gradient-test.log"
while IFS= read -r profile_manifest; do
    sha256sum -c "$profile_manifest"
done < <(rg --files "$run_path/source/profiles" | rg '/manifest.sha256$' | sort) > "$run_path/source-validation.log"
jq -n --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --slurpfile s "$summary_path" --slurpfile r "$run_path/geometry/rendering.json" '{
  state:"complete",completed_utc:$time,defense_continuations:8,attack_trajectories:36,
  regular_benchmark_points:162,default_local_searches:9,adaptive_fine_searches:1,
  geometry_cases:9,planes_per_case:5,grid_records_per_plane:81,
  geometry_all_case_ce_limits:$r[0].all_case_ce_limits,
  utility_passing_plot_cases:$r[0].utility_passing_cases,
  utility_passing_ce_limits:$r[0].utility_passing_ce_limits,
  matched_fidelity_states:($s[0].attacks|map(select(.matched==true))|group_by(.fidelity_state)|map({state:.[0].fidelity_state,n:length})),
  tests:{cpu_and_plot_methods_passed:70,multi_gpu_methods_passed_separately:2},
  interpretation:"Complete execution and evidence coverage; objective satisfaction, numerical fidelity, retained utility, and recovery remain separate outcomes."
}' > "$run_path/completion.json"
selection_list="$run_path/curated-files.txt"
{
    rg --files "$run_path" |
      rg -v '/(cache|__pycache__|models|revisions)/|/(defense|checkpoint|optimizer)-[0-9]+|/geometry-progress.json$|/(artifact-manifest.sha256|curated-files.txt|curated-pathspec.nul)$' |
      rg '\.(json|jsonl|md|csv|png|pdf|py|sh|yml|yaml|toml|lock|sha256)$|/(final-tests|gpu-cache-test|gpu-gradient-test|source-validation)\.log$'
    rg --files "$run_path/revisions/control-geometry-plot-snapshots" | rg '\.json$'
} | LC_ALL=C sort -u > "$selection_list"
manifest_path="$run_path/artifact-manifest.sha256"
printf '# Verify from examples/tamper-resistance. Model weights and optimizer states are excluded.\n' > "$manifest_path"
while IFS= read -r selected_file; do
    [[ $(stat -c %s "$selected_file") -lt 50000000 ]]
    sha256sum "$selected_file" | sed "s|  $PWD/|  |"
done < "$selection_list" >> "$manifest_path"
sha256sum -c "$manifest_path" > "$run_path/manifest-validation.log"
project_root=$(git rev-parse --show-toplevel)
stage_list="$run_path/curated-pathspec.nul"
while IFS= read -r selected_file; do
    printf '%s\0' "${selected_file#"$project_root"/}"
done < "$selection_list" > "$stage_list"
printf '%s\0' "${manifest_path#"$project_root"/}" >> "$stage_list"
printf 'Verified %s curated files.\n' "$(wc -l < "$selection_list")"
