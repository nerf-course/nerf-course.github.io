# USING NEW API: https://pyairtable.readthedocs.io/en/latest/getting-started.html
import os
from pathlib import Path
import pyairtable
import tqdm

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class View():
  def __init__(self, url:str, table:str, view:str):
    self._view=view
    api_key = self.get_airtable_api_key()
    base_id = self.get_base_id(url)
    self._table = pyairtable.Table(api_key, base_id, table)

  def get_base_id(self, url="https://airtable.com/appQTptcq51TTYJAc"):
    prefix = "https://airtable.com/"
    assert url.startswith(prefix)
    return url[len(prefix):]

  def get_airtable_api_key(self):
    try:
      return os.environ["AIRTABLE_API_KEY"]
    except KeyError:
      # TODO: mode="JSON" alike frank's
      print("ERROR: could not find environment variable 'AIRTABLE_API_KEY'.")

  def all(self, fields=None, formula=None):
    records=self._table.all(view=self._view, formula=formula, fields=fields)
    records=[record["fields"] for record in records]
    return records

  def get(self, record_id: str):
    return self._table.get(record_id)["fields"]

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

lorem = "At vero eos et accusamus et iusto odio dignissimos ducimus qui "\
        "blanditiis praesentium voluptatum deleniti atque corrupti quos "\
        "dolores et quas molestias excepturi sint occaecati cupiditate "\
        "non provident, similique sunt in culpa qui officia deserunt "\
        "mollitia animi, id est laborum et dolorum fuga."

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

url = "https://airtable.com/appQTptcq51TTYJAc"
topics_main = View(url, table="CourseTopics", view="Main")
papers_course = View(url, table="Papers", view="Course")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

output_path = Path("index.md")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as file:
  topics = topics_main.all()
  print("# Nerf Progression", file=file)
  #print("\{: .no_toc \}", file=file)
  page_num = 1
  for topic in tqdm.tqdm(topics):
    # TODO: could be done with filter, but not working?
    if "Exclude" in topic: continue

    # --- Print title and TL;DR for topic
    print(f"## {topic['Topic']}", file=file)
    toptitle = topic['Topic'].lower().replace(" ", "-")
    index_path = Path(f"{toptitle}/index.md")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as ifile:
      print("---", file=ifile)
      print("layout: forward", file=ifile)
      print(f"target: https://nerf-course.github.io/#{toptitle}", file=ifile)
      print(f"title: {topic['Topic']}", file=ifile)
      print(f"nav_order: {page_num}", file=ifile)
      page_num = page_num + 1
      print("---", file=ifile)
      print("hello", file=ifile)
    tldr = topic['TL;DR'] if "TL;DR" in topic else lorem
    print(f"{tldr}\n", file=file)

    # --- Fetch the papers in a topic
    paper_keys = topic["Papers"]
    papers = [papers_course.get(key) for key in paper_keys]

    # --- Prints paper content
    try:
      for ipaper, paper in enumerate(papers):
        # limit to 3 papers/topic
        short = paper['Short']
        if ipaper<3:
          # --- print title + links
          print(f"### [{short}]({paper['Project']}) @ {paper['VenueName']} {paper['Year']}", end=" – ", file=file)
          if 'Arxiv' in paper: print(f"[arXiv](https://arxiv.org/abs/{paper['Arxiv']})", end=" ", file=file)
          if 'YouTube' in paper: print(f"[YouTube]({paper['YouTube']})", end=" ", file=file)
          if 'Homepage' in paper: print(f"[Homepage]({paper['Homepage']})", end=" ", file=file)
          if 'TeaserURL' in paper: print(paper['Teaser'], file=file)
          # --- print TL;DR
          tldr = paper["TL;DR"] if "TL;DR" in paper else lorem
          print(f"\n{tldr}", end="\n\n", file=file)

    except BaseException as err:
      print(f"paper[{short}]: {type(err)} → {err}")
