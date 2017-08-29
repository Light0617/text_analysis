from urllib2 import urlopen
from lxml import etree
import re



def get_story(url):
	"""
	Given a url, we can get the story content, which is a list containing all paragraphs of the story.
	And each item is each paragraph of the story.
	@para : url
	@return : list	
	"""
	html = urlopen(url).read()
	content = etree.HTML(html)
	
	#get the story content to story_obj which is a list containing all paragraphs of the story and each item means each paragraph of the story
	story_obj = content.xpath(u"//div[contains(@class,'tabs-panel is-active')]//div[contains(@class, 'co-story')]/descendant::text()")
	#remove empty paragraph in story_obj
	story_obj = [x.strip() for x in story_obj if len(x.strip()) > 0]
	
	return ' '.join(story_obj)
	
if __name__  == "__main__":
	url = "https://www.gofundme.com/justin-reed-medical-fun"	
	url = "https://www.gofundme.com/mauricios-renteria-medical-founds"
	story = get_story(url)
	print story
