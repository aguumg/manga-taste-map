import json
sc = json.load(open('data/western_search_cache.json'))
for q in ['Crossed::12', 'Sara::12', 'Y The Last Man::12', 'Saga::12',
          'Crecy::12', 'Locke & Key::12']:
    hits = sc.get(q, [])
    label = q.split('::')[0]
    print(label, '->', [(h['id'], h['name'], h['publisher'], h['start_year'])
                        for h in hits[:3]])
