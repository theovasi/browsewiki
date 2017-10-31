# -*- coding: utf-8 -*-
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy import Item, Field


class Document(Item):
    title = Field()


class DisambSpider(CrawlSpider):
    """ Spider that scrapes article titles from the disambiguation category of the 
        Greek Wikipedia.
    """

    name = 'disamb'
    custom_settings = {
        'ITEM_PIPELINES': {
            'wikicrawlers.pipelines.PicklePipeline': 300
        }
    }
    allowed_domains = ['el.wikipedia.org']
    start_urls = ['https://el.wikipedia.org/wiki/%CE%9A%CE%B1%CF%84%CE%B7%CE%B3%CE%BF%CF%81%CE%AF%CE%B1:%CE%91%CF%80%CE%BF%CF%83%CE%B1%CF%86%CE%AE%CE%BD%CE%B9%CF%83%CE%B7']

    rules = (
        # Link to next page.
        Rule(LinkExtractor(allow='\/w\/index\.php\?title=%CE%9A%CE%B1%CF%84%CE%B7%CE%B3%CE%BF%CF%81%CE%AF%CE%B1:%CE%91%CF%80%CE%BF%CF%83%CE%B1%CF%86%CE%AE%CE%BD%CE%B9%CF%83%CE%B7&pagefrom='), callback="parse_item", follow=True),)


    def parse_item(self, response):
        disamb_titles = (response.css(
                         '.mw-category > .mw-category-group > ul > li > a::attr(title)').extract())
        for title in disamb_titles:
            doc = Document()
            doc['title'] = title
            yield doc

