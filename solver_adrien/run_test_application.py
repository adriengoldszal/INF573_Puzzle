from application import run_realtime_view

url = "http://10.220.14.33:8080/video"
puzzle_image_path = "nos_puzzles/yakari.jpg"
verbose = False
update_interval=10
# Run the real-time view
run_realtime_view(url, puzzle_image_path, update_interval, verbose)