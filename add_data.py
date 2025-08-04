# Download the Youtube transcript and 

import argparse
from pprint import pprint

from local_knowledge.youtube_transcripts import YouTubeTranscriptProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download the transcript from a YouTube video and store it in a MongoDB and chromadb (vector database)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--url_or_id", type=str, required=True, help="YouTube video URL or ID")
    parser.add_argument("--ads_start_time", type=int, default=0, help="Start time of ads in seconds")
    parser.add_argument("--ads_end_time", type=int, default=0, help="End time of ads in seconds")
    parser.add_argument("--transcript_end_time", type=float, default=float('inf'), help="End time of transcript in seconds")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing transcript")
    parser.add_argument("--enable_vector_storage", action="store_true", default=True, help="Enable vector storage")
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose output")

    args = parser.parse_args()

    processor = YouTubeTranscriptProcessor()

    video_url = args.url_or_id
    ads_times = (args.ads_start_time, args.ads_end_time)
    end_time = args.transcript_end_time

    transcript_doc_data = processor.fetch_and_store_transcript(
        video_url_or_id=video_url,
        ads_times=ads_times,
        end_time=end_time,
        overwrite=args.overwrite,
        enable_vector_storage=args.enable_vector_storage
    )

    if args.verbose:
        pprint(transcript_doc_data)
