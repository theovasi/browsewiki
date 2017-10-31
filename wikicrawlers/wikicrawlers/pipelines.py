# -*- coding: utf-8 -*-
import os
import joblib


class PicklePipeline(object):
    def open_spider(self, spider):
        self.item_list = []

    def close_spider(self, spider):
        if os.path.exists('ignore_titles.pkl'):
            titles = joblib.load('ignore_titles.pkl')
            titles.extend(self.item_list)
            joblib.dump(titles, 'ignore_titles.pkl')
        else:
            joblib.dump(self.item_list, 'ignore_titles.pkl')

    def process_item(self, item, spider):
        self.item_list.append(item['title'])
        return item
