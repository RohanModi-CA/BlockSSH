_blocksssh_catalog() {
    local kind="$1"
    shift || true
    "$BLOCKSSSH_PYTHON_BIN" "$BLOCKSSSH_ROOT/scripts/catalog_cli.py" "$kind" "$@" 2>/dev/null
}

_blocksssh_cached_catalog() {
    local key="$1"
    printf '%s' "${BLOCKSSSH_COMPLETION_CATALOGS["$key"]}"
}

_blocksssh_option_list() {
    local script_key="$1"
    printf '%s' "${BLOCKSSSH_COMPLETION_OPTIONS["$script_key"]}"
}

_blocksssh_value_option_list() {
    local script_key="$1"
    printf '%s' "${BLOCKSSSH_COMPLETION_VALUE_OPTIONS["$script_key"]}"
}

_blocksssh_option_choices() {
    local script_key="$1"
    local option="$2"
    printf '%s' "${BLOCKSSSH_COMPLETION_CHOICES["$script_key::$option"]}"
}

_blocksssh_option_takes_value() {
    local script_key="$1"
    local option="$2"
    local options=" $(_blocksssh_value_option_list "$script_key") "
    [[ "$options" == *" $option "* ]]
}

_blocksssh_compgen() {
    local cur="$1"
    shift || true
    COMPREPLY=( $(compgen -W "$*" -- "$cur") )
}

_blocksssh_reply_from_values() {
    local cur="$1"
    shift || true
    local values="$*"
    COMPREPLY=( $(compgen -W "$values" -- "$cur") )
}

_blocksssh_reply_from_catalog() {
    local cur="$1"
    local kind="$2"
    local dataset="${3:-}"
    local cache_key="$kind"
    local values=""
    if [[ "$kind" == "components" && -n "$dataset" ]]; then
        cache_key="components::$dataset"
    fi
    values="$(_blocksssh_cached_catalog "$cache_key")"
    if [[ -z "$values" ]]; then
        if [[ "$kind" == "components" && -n "$dataset" ]]; then
            values="$(_blocksssh_catalog "$kind" "$dataset" | tr '\n' ' ')"
        else
            values="$(_blocksssh_catalog "$kind" | tr '\n' ' ')"
        fi
    fi
    COMPREPLY=( $(compgen -W "$values" -- "$cur") )
}

_blocksssh_script_key() {
    local word="$1"
    case "$word" in
        FFT.py|*analysis/go/FFT.py) echo "analysis/go/FFT.py" ;;
        Timeseries.py|*analysis/go/Timeseries.py) echo "analysis/go/Timeseries.py" ;;
        Subtract.py|*analysis/go/Subtract.py) echo "analysis/go/Subtract.py" ;;
        Wavefunctions.py|*analysis/go/Wavefunctions.py) echo "analysis/go/Wavefunctions.py" ;;
        ClickPeakFind.py|*analysis/go/ClickPeakFind.py) echo "analysis/go/ClickPeakFind.py" ;;
        SpectrasaveView.py|*analysis/go/SpectrasaveView.py) echo "analysis/go/SpectrasaveView.py" ;;
        MakeGroup.py|*analysis/go/MakeGroup.py) echo "analysis/go/MakeGroup.py" ;;
        0.VideoPrepareBottom.py|*track/Bottom/0.VideoPrepareBottom.py) echo "track/Bottom/0.VideoPrepareBottom.py" ;;
        1.TrackRun.py|*track/Bottom/1.TrackRun.py) echo "track/Bottom/1.TrackRun.py" ;;
        2.ProcessVerify.py|*track/Bottom/2.ProcessVerify.py) echo "track/Bottom/2.ProcessVerify.py" ;;
        2b.ManualRepair.py|*track/Bottom/2b.ManualRepair.py) echo "track/Bottom/2b.ManualRepair.py" ;;
        3.Label.py|*track/Bottom/3.Label.py) echo "track/Bottom/3.Label.py" ;;
        B0.BatchPrepare.py|*track/Bottom/B0.BatchPrepare.py) echo "track/Bottom/B0.BatchPrepare.py" ;;
        B1.BatchTrack.py|*track/Bottom/B1.BatchTrack.py) echo "track/Bottom/B1.BatchTrack.py" ;;
        B2.BatchProcessVerify.py|*track/Bottom/B2.BatchProcessVerify.py) echo "track/Bottom/B2.BatchProcessVerify.py" ;;
        *analysis/go/FFT.py) echo "analysis/go/FFT.py" ;;
        *analysis/go/Timeseries.py) echo "analysis/go/Timeseries.py" ;;
        *analysis/go/Subtract.py) echo "analysis/go/Subtract.py" ;;
        *analysis/go/Wavefunctions.py) echo "analysis/go/Wavefunctions.py" ;;
        *analysis/go/ClickPeakFind.py) echo "analysis/go/ClickPeakFind.py" ;;
        *analysis/go/SpectrasaveView.py) echo "analysis/go/SpectrasaveView.py" ;;
        *analysis/go/MakeGroup.py) echo "analysis/go/MakeGroup.py" ;;
        *track/Bottom/0.VideoPrepareBottom.py) echo "track/Bottom/0.VideoPrepareBottom.py" ;;
        *track/Bottom/1.TrackRun.py) echo "track/Bottom/1.TrackRun.py" ;;
        *track/Bottom/2.ProcessVerify.py) echo "track/Bottom/2.ProcessVerify.py" ;;
        *track/Bottom/2b.ManualRepair.py) echo "track/Bottom/2b.ManualRepair.py" ;;
        *track/Bottom/3.Label.py) echo "track/Bottom/3.Label.py" ;;
        *track/Bottom/B0.BatchPrepare.py) echo "track/Bottom/B0.BatchPrepare.py" ;;
        *track/Bottom/B1.BatchTrack.py) echo "track/Bottom/B1.BatchTrack.py" ;;
        *track/Bottom/B2.BatchProcessVerify.py) echo "track/Bottom/B2.BatchProcessVerify.py" ;;
        *track/0.video_prepare_black.py) echo "track/Bottom/0.VideoPrepareBottom.py" ;;
        *track/1.track_run_black.py) echo "track/Bottom/1.TrackRun.py" ;;
        *track/2.verify_and_process_black.py) echo "track/Bottom/2.ProcessVerify.py" ;;
        *track/2b.repair_unrepairable_black.py) echo "track/Bottom/2b.ManualRepair.py" ;;
        *) echo "" ;;
    esac
}

_blocksssh_positional_index_after_script() {
    local script_key="$1"
    local script_index="$2"
    local idx
    local count=0
    for (( idx=script_index+1; idx < COMP_CWORD; idx++ )); do
        if [[ "${COMP_WORDS[idx]}" == -* ]]; then
            if _blocksssh_option_takes_value "$script_key" "${COMP_WORDS[idx]}"; then
                ((idx++))
            fi
            continue
        fi
        ((count++))
    done
    echo "$count"
}

_blocksssh_complete_by_script() {
    local script_key="$1"
    local script_index="$2"
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local positional_index
    positional_index="$(_blocksssh_positional_index_after_script "$script_key" "$script_index")"

    if [[ "$cur" == -* ]]; then
        _blocksssh_reply_from_values "$cur" $(_blocksssh_option_list "$script_key")
        return 0
    fi

    local choice_values
    choice_values="$(_blocksssh_option_choices "$script_key" "$prev")"
    if [[ -n "$choice_values" ]]; then
        _blocksssh_reply_from_values "$cur" $choice_values
        return 0
    fi

    case "$prev" in
        --group)
            _blocksssh_reply_from_catalog "$cur" groups
            return 0
            ;;
        --exclude)
            case "$script_key" in
                track/Bottom/B1.BatchTrack.py|track/Bottom/B0.BatchPrepare.py)
                    _blocksssh_reply_from_catalog "$cur" prepare-targets
                    return 0
                    ;;
                track/Bottom/B2.BatchProcessVerify.py)
                    _blocksssh_reply_from_catalog "$cur" datasets
                    return 0
                    ;;
            esac
            ;;
    esac

    case "$script_key" in
        analysis/go/FFT.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" datasets
                return 0
            fi
            ;;
        analysis/go/Timeseries.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" datasets
                return 0
            fi
            ;;
        analysis/go/Subtract.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" datasets
                return 0
            fi
            ;;
        analysis/go/Wavefunctions.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" datasets
                return 0
            fi
            if [[ "$positional_index" -eq 1 ]]; then
                _blocksssh_reply_from_catalog "$cur" peaks
                return 0
            fi
            ;;
        analysis/go/ClickPeakFind.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" spectrasaves
                return 0
            fi
            if [[ "$positional_index" -eq 1 ]]; then
                _blocksssh_reply_from_catalog "$cur" peaks
                return 0
            fi
            ;;
        analysis/go/SpectrasaveView.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" spectrasaves
                return 0
            fi
            ;;
        analysis/go/MakeGroup.py)
            _blocksssh_reply_from_catalog "$cur" datasets
            return 0
            ;;
        track/Bottom/0.VideoPrepareBottom.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" prepare-targets
                return 0
            fi
            ;;
        track/Bottom/1.TrackRun.py|track/Bottom/2.ProcessVerify.py|track/Bottom/2b.ManualRepair.py|track/Bottom/3.Label.py)
            if [[ "$positional_index" -eq 0 ]]; then
                _blocksssh_reply_from_catalog "$cur" datasets
                return 0
            fi
            ;;
        track/Bottom/B0.BatchPrepare.py|track/Bottom/B1.BatchTrack.py|track/Bottom/B2.BatchProcessVerify.py)
            COMPREPLY=()
            return 1
            ;;
    esac

    COMPREPLY=()
    return 1
}

_blocksssh_python_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local script_key
    script_key="$(_blocksssh_script_key "${COMP_WORDS[1]}")"
    if [[ -n "$script_key" ]]; then
        _blocksssh_complete_by_script "$script_key" 1 && return 0
    fi
    COMPREPLY=()
    return 0
}

_blocksssh_direct_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local script_key
    script_key="$(_blocksssh_script_key "${COMP_WORDS[0]}")"
    if [[ -n "$script_key" ]]; then
        _blocksssh_complete_by_script "$script_key" 0 && return 0
    fi
    COMPREPLY=()
    return 0
}

_blocksssh_function_completion() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local key=""
    case "${COMP_WORDS[0]}" in
        bfft) key="analysis/go/FFT.py" ;;
        bts) key="analysis/go/Timeseries.py" ;;
        bsub) key="analysis/go/Subtract.py" ;;
        bwf) key="analysis/go/Wavefunctions.py" ;;
        bpeak) key="analysis/go/ClickPeakFind.py" ;;
        bspec) key="analysis/go/SpectrasaveView.py" ;;
        bgroup) key="analysis/go/MakeGroup.py" ;;
        bprep) key="track/Bottom/0.VideoPrepareBottom.py" ;;
        brun) key="track/Bottom/1.TrackRun.py" ;;
        bproc) key="track/Bottom/2.ProcessVerify.py" ;;
        brepair) key="track/Bottom/2b.ManualRepair.py" ;;
        blabel) key="track/Bottom/3.Label.py" ;;
        bprepbatch) key="track/Bottom/B0.BatchPrepare.py" ;;
        btrackbatch) key="track/Bottom/B1.BatchTrack.py" ;;
        bprocbatch) key="track/Bottom/B2.BatchProcessVerify.py" ;;
    esac
    if [[ -n "$key" ]]; then
        _blocksssh_complete_by_script "$key" 0 && return 0
    fi
    COMPREPLY=()
}

complete -o default -F _blocksssh_python_completion python3
complete -o default -F _blocksssh_python_completion python
complete -o default -F _blocksssh_direct_completion \
    analysis/go/FFT.py \
    analysis/go/Timeseries.py \
    analysis/go/Subtract.py \
    analysis/go/Wavefunctions.py \
    analysis/go/ClickPeakFind.py \
    analysis/go/SpectrasaveView.py \
    analysis/go/MakeGroup.py \
    track/Bottom/0.VideoPrepareBottom.py \
    track/Bottom/1.TrackRun.py \
    track/Bottom/2.ProcessVerify.py \
    track/Bottom/2b.ManualRepair.py \
    track/Bottom/3.Label.py \
    track/Bottom/B0.BatchPrepare.py \
    track/Bottom/B1.BatchTrack.py \
    track/Bottom/B2.BatchProcessVerify.py \
    ./analysis/go/FFT.py \
    ./analysis/go/Timeseries.py \
    ./analysis/go/Subtract.py \
    ./analysis/go/Wavefunctions.py \
    ./analysis/go/ClickPeakFind.py \
    ./analysis/go/SpectrasaveView.py \
    ./analysis/go/MakeGroup.py \
    ./track/Bottom/0.VideoPrepareBottom.py \
    ./track/Bottom/1.TrackRun.py \
    ./track/Bottom/2.ProcessVerify.py \
    ./track/Bottom/2b.ManualRepair.py \
    ./track/Bottom/3.Label.py \
    ./track/Bottom/B0.BatchPrepare.py \
    ./track/Bottom/B1.BatchTrack.py \
    ./track/Bottom/B2.BatchProcessVerify.py \
    "$BLOCKSSSH_ROOT/analysis/go/FFT.py" \
    "$BLOCKSSSH_ROOT/analysis/go/Timeseries.py" \
    "$BLOCKSSSH_ROOT/analysis/go/Subtract.py" \
    "$BLOCKSSSH_ROOT/analysis/go/Wavefunctions.py" \
    "$BLOCKSSSH_ROOT/analysis/go/ClickPeakFind.py" \
    "$BLOCKSSSH_ROOT/analysis/go/SpectrasaveView.py" \
    "$BLOCKSSSH_ROOT/analysis/go/MakeGroup.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/0.VideoPrepareBottom.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/1.TrackRun.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/2.ProcessVerify.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/2b.ManualRepair.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/3.Label.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/B0.BatchPrepare.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/B1.BatchTrack.py" \
    "$BLOCKSSSH_ROOT/track/Bottom/B2.BatchProcessVerify.py"
complete -o default -F _blocksssh_function_completion \
    bfft bts bsub bwf bpeak bspec bgroup bprep brun bproc brepair blabel \
    bprepbatch btrackbatch bprocbatch
