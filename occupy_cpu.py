import time


def main():
    print("Starting a process to keep one CPU occupied.")
    while True:
        for _ in range(100000000):
            x = 1
            y = 2
            z = x + y
        time.sleep(0.01)
        print("Running again.")

if __name__ == '__main__':
    main()
