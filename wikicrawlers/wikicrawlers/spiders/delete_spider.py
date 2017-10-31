# -*- coding: utf-8 -*-
from scrapy import Item, Field, Spider


class Document(Item):
    title = Field()

class DeleteSpider(Spider):
    name = 'delete'
    custom_settings = {
        'ITEM_PIPELINES': {
            'wikicrawlers.pipelines.PicklePipeline': 300
        }
    }
    allowed_domains = ['el.wikipedia.org']
    start_urls = ['https://el.wikipedia.org/wiki/%CE%9A%CE%B1%CF%84%CE%B7%CE%B3%CE%BF%CF%81%CE%AF%CE%B1:%CE%A3%CE%B5%CE%BB%CE%AF%CE%B4%CE%B5%CF%82_%CE%B3%CE%B9%CE%B1_%CE%B3%CF%81%CE%AE%CE%B3%CE%BF%CF%81%CE%B7_%CE%B4%CE%B9%CE%B1%CE%B3%CF%81%CE%B1%CF%86%CE%AE']


    def parse(self, response):
        delete_titles = response.css(
                    '.mw-category > .mw-category-group > ul > li > a::attr(title)').extract()
        for title in delete_titles:
            doc = Document()
            doc['title'] = title
            yield doc 

