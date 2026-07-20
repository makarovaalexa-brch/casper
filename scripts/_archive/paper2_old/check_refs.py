import re
t=open('C:/dev/phd/papers/paper2_casper/paper2_casper.tex',encoding='utf-8').read()
labels=set(re.findall(r'\\label\{([^}]+)\}', t))
refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', t))
bibs=set(re.findall(r'\\bibitem\{([^}]+)\}', t))
cites=set()
for c in re.findall(r'\\cite\{([^}]+)\}', t):
    for k in c.split(','): cites.add(k.strip())
print('REFS missing labels:', sorted(refs-labels) or 'OK')
print('CITES missing bibitems:', sorted(cites-bibs) or 'OK')
print('unused bibitems:', sorted(bibs-cites))
print('counts -> tables:', len(re.findall(r'\\begin\{table\}',t)),
      'figures:', len(re.findall(r'\\begin\{figure\}',t)),
      'algorithms:', len(re.findall(r'\\begin\{algorithm\}',t)),
      'sections:', len(re.findall(r'\\section\{',t)))
body=re.sub(r'%.*','',t); body=re.sub(r'\\[a-zA-Z]+\*?','',body); body=re.sub(r'[{}\\]',' ',body)
print('approx words:', len(re.findall(r'\b[A-Za-z]{2,}\b',body)))
