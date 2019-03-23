from flickrapi import FlickrAPI
import pandas as pd


key = 'c81a9c03a2e576a2ca737cd48547c9ef'
secret = '62faf5cdfb2c8094'


def get_urls(image_tag, MAX_COUNT):
    flickr = FlickrAPI(key, secret)
    photos = flickr.walk(text=image_tag,
                         tag_mode='all',
                         tags=image_tag,
                         extras='url_o',
                         per_page=50,
                         sort='relevance'
                         )
    count = 0
    urls = []
    for photo in photos:
        if count < MAX_COUNT:
            count = count+1
            print("Fetching url for image number {}".format(count))
            try:
                url = photo.get('url_o')
                urls.append(url)
            except:
                print("Url for image number {} could not be fetched".format(count))
        else:
            print("Done fetching urls, fetched {} urls out of {}".format(len(urls),MAX_COUNT))
            break
    urls = pd.Series(urls)
    print("Writing out the urls in the current directory")
    urls.to_csv(image_tag+"_urls.csv")
    print("Done!!!")


if __name__ == '__main__':
    get_urls('great white shark', 1000)
    get_urls('hammerhead shark', 1000)
