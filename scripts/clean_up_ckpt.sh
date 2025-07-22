RESULT_DIR="/home/zhaochengz/lustre/reinforcer/results"

for checkpoint_dir in ${RESULT_DIR}/*; do
    model_family=$(basename ${checkpoint_dir})
    summary_files=(
        "logs/${model_family}_greedy_summary.txt"
        "logs/${model_family}_recommended_summary.txt"
        "logs/${model_family}_high_summary.txt"
    )
    hf_dirs=$(ls -d ${checkpoint_dir}/hf_step_* 2>/dev/null | sort -V)

    dirs_to_remove=()
    for hf_dir in ${hf_dirs}; do
        record="model_name='$(basename $hf_dir)'"
        keep=false
        for summary_file in ${summary_files[@]}; do
            if [ ! -f "${summary_file}" ] || ! grep -q "${record}" "${summary_file}"; then
                keep=true
                break
            fi
        done
        if [ "${keep}" = false ]; then
            dirs_to_remove+=(${hf_dir})
        fi
    done

    if [ ${#dirs_to_remove[@]} -gt 0 ]; then
        echo "${checkpoint_dir}/"
        for dir in ${dirs_to_remove[@]}; do
            echo "    $(basename ${dir})"
        done
        read -p "Remove ${#dirs_to_remove[@]} checkpoints? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf ${dirs_to_remove[@]}
        fi
    fi
done