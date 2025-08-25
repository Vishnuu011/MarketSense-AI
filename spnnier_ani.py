import sys
import time
import itertools
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Color dictionary
COLORS = {
    "green": Fore.GREEN,
    "red": Fore.RED,
    "purple": Fore.MAGENTA,
    "blue": Fore.BLUE,
    "yellow": Fore.YELLOW,
    "cyan": Fore.CYAN,
    "white": Fore.GREEN
}


SPINNERS = {
    "input": itertools.cycle(["◐", "◓", "◑", "◒"]),
    "train_predict": itertools.cycle(["⠋","⠙","⠸","⠴","⠦","⠧"]),
    "delete_model": itertools.cycle(["✖", "☓", "✗", "❌"]),
    "market_analysis": itertools.cycle(["◆", "◇", "◈", "⬢"]),
    "exit": itertools.cycle(["✔", "☑", "✓", "✅"]),
}

def spinner(task: str, duration: float = 3, color: str = "white"):
    """
    Display spinner animation based on task type and color.
    """
    spinner_cycle = SPINNERS.get(task, SPINNERS["input"])  # default spinner
    chosen_color = COLORS.get(color.lower(), Fore.WHITE)   # default color

    start_time = time.time()
    while time.time() - start_time < duration:
        sys.stdout.write(f"\r{chosen_color}Loading {next(spinner_cycle)}{Style.RESET_ALL}")
        sys.stdout.flush()
        time.sleep(0.1)

    sys.stdout.write(f"\r{chosen_color}Done! ✔{Style.RESET_ALL}\n")



if __name__ == "__main__":
    print("User entering input...")
    spinner("input", 2, color="cyan")

    print("Training RL model + prediction...")
    spinner("train_predict", 3, color="green")

    print("Deleting previous RL model...")
    spinner("delete_model", 2, color="red")

    print("Running Market Analysis with CrewAI...")
    spinner("market_analysis", 4, color="purple")

    print("Exiting program...")
    spinner("exit", 2, color="yellow")
