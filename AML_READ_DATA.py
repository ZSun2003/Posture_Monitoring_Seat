import time
import sys
import threading
import queue
import shlex
import serial
from pathlib import Path

# ---------- 平台切换(Win / macOS) ----------
OS_MODE = "mac"  # "win" / "mac"

if OS_MODE.lower() == "win":
    DEFAULT_SERIAL_PORT = "COM3"
    DEFAULT_OUTPUT_DIR = r"C:\Users\17717\Desktop\AML_DATASET\Sitfront"
elif OS_MODE.lower() == "mac":
    DEFAULT_SERIAL_PORT = "/dev/cu.usbmodem1101"
    DEFAULT_OUTPUT_DIR = "~/Desktop/AML_DATASET/Sitfront"
else:
    raise ValueError("OS_MODE 只能是 'win' 或 'mac'")

CONFIG = {
    "port": DEFAULT_SERIAL_PORT,
    "baud": 115200,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "filename": "participant_4.csv",
    "overwrite": True,
}

# =========================================================

def open_output_file(cfg, cached_header=None):
    out_dir = Path(cfg["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / cfg["filename"]

    if path.exists() and not cfg["overwrite"]:
        raise FileExistsError(f"File exists and overwrite=False: {path}")

    f = open(path, "w", encoding="utf-8", newline="")
    if cached_header:
        f.write(cached_header + "\n")
        f.flush()
    return f, path


def serial_send_line(ser: serial.Serial, s: str):
    # Mega: 以 '/' 或 '\n' 结束；这里统一用 '/'
    msg = (s.strip().upper() + "/").encode("utf-8", errors="ignore")
    ser.write(msg)
    ser.flush()



def stdin_worker(cmd_q: queue.Queue):
    print("\n[CMD] 输入命令然后回车：")
    print("  c / cal        -> 发送 CAL (用于 ToF 三步校准；需要按提示发 3 次)")
    print("  ping           -> 发送 PING")
    print("  file xxx.csv   -> 切换输出文件名（同目录）")
    print("  dir  <path>    -> 切换输出目录")
    print("  quit           -> 退出采集\n")
    while True:
        try:
            s = input().strip()
        except EOFError:
            return
        cmd_q.put(s)


def is_csv_data_line(line: str) -> bool:
    # header: t_ms,...
    if line.startswith("t_ms"):
        return True
    # 数据行：一般以数字开头（t_ms），且包含逗号
    if "," in line and len(line) > 0 and (line[0].isdigit() or line[0] == "-"):
        return True
    return False


def main():
    # 打开串口（Arduino IDE 的 Serial Monitor 必须关闭）
    ser = serial.Serial(CONFIG["port"], CONFIG["baud"], timeout=0.5)
    time.sleep(2)  # 等 Arduino 自动复位（会重新进入 ToF 校准流程）

    # 可选：清掉刚打开时的残留缓冲（校准会阻塞等待 CAL，所以不会丢 header）
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    cmd_q: queue.Queue[str] = queue.Queue()
    threading.Thread(target=stdin_worker, args=(cmd_q,), daemon=True).start()

    cached_header = None
    header_written = False

    out_f, out_path = open_output_file(CONFIG, cached_header=None)

    print("Serial connected:", CONFIG["port"])
    print("Writing to:", out_path)

    try:
        while True:
            # =========================
            # 1) 处理键盘命令
            # =========================
            while not cmd_q.empty():
                raw = cmd_q.get()
                if not raw:
                    continue

                parts = shlex.split(raw)
                cmd = parts[0].lower()

                if cmd == "quit":
                    print("[OK] quit requested.")
                    return

                if cmd in ("cal", "c"):
                    serial_send_line(ser, "CAL")
                    print("[UART TX] CAL")
                    continue


                if cmd == "ping":
                    serial_send_line(ser, "PING")
                    print("[UART TX] PING")
                    continue

                if cmd == "file" and len(parts) >= 2:
                    CONFIG["filename"] = parts[1]
                    out_f.flush()
                    out_f.close()
                    out_f, out_path = open_output_file(CONFIG, cached_header=cached_header)
                    header_written = cached_header is not None
                    print("[OK] switched file ->", out_path)
                    continue

                if cmd == "dir" and len(parts) >= 2:
                    CONFIG["output_dir"] = parts[1]
                    out_f.flush()
                    out_f.close()
                    out_f, out_path = open_output_file(CONFIG, cached_header=cached_header)
                    header_written = cached_header is not None
                    print("[OK] switched dir  ->", out_path)
                    continue

                # 其它命令：原样转发给 Mega（例如你以后扩展命令）
                serial_send_line(ser, raw)
                print("[UART TX]", raw)

            # =========================
            # 2) 读取串口
            # =========================
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # 终端原样打印
            print(line)

            # 调试行不写入 CSV
            if line.startswith("#"):
                continue

            # 只处理 CSV header 或 数据行
            if not is_csv_data_line(line):
                continue

            # header
            if line.startswith("t_ms"):
                cached_header = line
                if not header_written:
                    out_f.write(line + "\n")
                    out_f.flush()
                    header_written = True
                continue

            # data line
            if "," in line:
                # 如果先出现了数据行但 header 还没写（极少数情况），补写 header
                if (not header_written) and cached_header:
                    out_f.write(cached_header + "\n")
                    header_written = True
                if header_written:
                    out_f.write(line + "\n")
                    out_f.flush()

    finally:
        try:
            out_f.flush()
            out_f.close()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
