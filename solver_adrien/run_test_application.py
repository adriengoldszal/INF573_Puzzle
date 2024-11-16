from application import run_realtime_view

url = "http://192.168.1.15:8080/video"
puzzle_image_path = "nos_puzzles/fete.jpg"
verbose = False
update_interval=10
# Run the real-time view
run_realtime_view(url, puzzle_image_path, update_interval, verbose)