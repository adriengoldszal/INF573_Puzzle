from application import run_realtime_view

#url = "http://192.168.225.205:8080/video"
url = "http://10.220.14.33:8080/video"
puzzle_image_path = "puzzles/chateau.jpg"
verbose = False
update_interval=20

run_realtime_view(url, puzzle_image_path, update_interval, verbose)