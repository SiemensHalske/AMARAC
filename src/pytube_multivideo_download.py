import os
import pytube
from pytube import YouTube
from tqdm import tqdm


def download_video(url, path):
    yt = YouTube(url)
    try:
        stream = yt.streams.get_highest_resolution()
    except Exception as e:
        print(e)
        stream = yt.streams.first()

    if stream:
        stream.download(path, filename=yt.title, 
                        filename_prefix='downloading: ')
    else:
        print("Could not find a valid stream.")


def main():
    """Main function."""

    url = input("Enter YouTube video or playlist URL: ")
    path = "C:\\Users\\Hendrik.Siemens\\Documents\\GitHub\\AMARAC\\data\\dataset_video"

    if not os.path.exists(path):
        os.makedirs(path)

    if "playlist" in url:
        playlist = pytube.Playlist(url)
        for video in tqdm(playlist.videos, desc="Downloading playlist"):
            download_video(video.watch_url, path)
    else:
        download_video(url, path)


if __name__ == "__main__":
    main()
