"""
Checks that flow CSVs from both CICFlowMeter implementations map onto every model feature.
The column names below are copied from each tool's source code:
  Java  : ahlashkari/CICFlowMeter  src/main/java/cic/cs/unb/ca/jnetpcap/FlowFeature.java
  Python: hieulw/cicflowmeter      src/cicflowmeter/flow.py (Flow.get_data keys)
Run:  python -m pytest tests/      (or simply: python tests/test_flow_columns.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flow_columns import DERIVE, _FEAT_LK, _key  # noqa: E402

FEATURES = json.loads((ROOT / "results" / "feature_list.json").read_text())

JAVA_V4 = [
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
    "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
PYTHON = (
    "flow_duration flow_byts_s flow_pkts_s fwd_pkts_s bwd_pkts_s tot_fwd_pkts tot_bwd_pkts "
    "totlen_fwd_pkts totlen_bwd_pkts fwd_pkt_len_max fwd_pkt_len_min fwd_pkt_len_mean "
    "fwd_pkt_len_std bwd_pkt_len_max bwd_pkt_len_min bwd_pkt_len_mean bwd_pkt_len_std pkt_len_max "
    "pkt_len_min pkt_len_mean pkt_len_std pkt_len_var fwd_header_len bwd_header_len "
    "fwd_seg_size_min fwd_act_data_pkts flow_iat_mean flow_iat_max flow_iat_min flow_iat_std "
    "fwd_iat_tot fwd_iat_max fwd_iat_min fwd_iat_mean fwd_iat_std bwd_iat_tot bwd_iat_max "
    "bwd_iat_min bwd_iat_mean bwd_iat_std fwd_psh_flags bwd_psh_flags fwd_urg_flags bwd_urg_flags "
    "fin_flag_cnt syn_flag_cnt rst_flag_cnt psh_flag_cnt ack_flag_cnt urg_flag_cnt ece_flag_cnt "
    "down_up_ratio pkt_size_avg init_fwd_win_byts init_bwd_win_byts active_max active_min "
    "active_mean active_std idle_max idle_min idle_mean idle_std fwd_byts_b_avg fwd_pkts_b_avg "
    "bwd_byts_b_avg bwd_pkts_b_avg fwd_blk_rate_avg bwd_blk_rate_avg"
).split()


def _mapped(names):
    return [_FEAT_LK[_key(n)] for n in names if _key(n) in _FEAT_LK]


def test_java_v4_covers_every_feature():
    assert set(FEATURES) <= set(_mapped(JAVA_V4)), set(FEATURES) - set(_mapped(JAVA_V4))


def test_python_port_covers_every_feature_except_derivable_ones():
    missing = set(FEATURES) - set(_mapped(PYTHON))
    assert missing <= set(DERIVE), missing - set(DERIVE)


def test_no_two_source_columns_map_to_the_same_feature():
    for names in (JAVA_V4, PYTHON):
        m = _mapped(names)
        assert len(m) == len(set(m))


if __name__ == "__main__":
    for t in (test_java_v4_covers_every_feature, test_python_port_covers_every_feature_except_derivable_ones,
              test_no_two_source_columns_map_to_the_same_feature):
        t()
        print("PASS", t.__name__)
