import datetime

class SimpleLogger:
    def __init__(self, class_name: str, level: str = "info"):
        self.class_name = class_name
        self.level = level.lower()
        self.levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    
    def _should_log(self, msg_level: str):
        return self.levels[msg_level] >= self.levels.get(self.level, 20)
    
    def _log(self, msg_level: str, message: str):
        if self._should_log(msg_level):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{msg_level.upper()}] [{self.class_name}] {message}")
    
    def debug(self, message: str):
        self._log("debug", message)
    
    def info(self, message: str):
        self._log("info", message)
    
    def warning(self, message: str):
        self._log("warning", message)
    
    def error(self, message: str):
        self._log("error", message)
