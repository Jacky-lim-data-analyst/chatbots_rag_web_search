# download YouTube Transcript and store it on local mongodb

import sys
import os
import argparse
from pprint import pprint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_knowledge.youtube_transcripts import YouTubeTranscriptProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YouTube transcript and store it in local MongoDB and ChromaDB."
    )
    parser.add_argument(
        "video_url",
        help="YouTube video URL or id"
    )
    parser.add_argument(
        "--ads-times",
        type=str,
        default=None,
        help="Comma-separated start,end times for ads to skip (e.g. 50,130)"
    )
    parser.add_argument(
        "--end-time",
        type=int,
        default=None,
        help="End-time in seconds for the transcript"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwriting existing transcript in database"
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    processor = YouTubeTranscriptProcessor()
    # modify the parameters below to import youtube transcripts
    
    if args.ads_times:
        try:
            start, end = map(int, args.ads_times.split(","))
            ads_times = (start, end)

        except Exception:
            print("Invalid --ads-times format. Use: start,end (e.g. 55,202)")
            ads_times = (0, 0)
            sys.exit(1)
    else:
        ads_times = (0, 0)

    transcript_doc_data = processor.fetch_and_store_transcript(
        video_url_or_id=args.video_url,
        ads_times=ads_times,
        end_time=args.end_time,
        overwrite=args.overwrite
    )

    pprint(transcript_doc_data)