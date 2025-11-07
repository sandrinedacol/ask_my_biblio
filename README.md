# Ask My Biblio

## How to run this amazing app?

### 1. Copy repo :
```
git clone git@github.com:sandrinedacol/biblio_finder.git
cd biblio_finder
```
### 2. Install dependencies :
```
pip install -r requirements.txt
```
### 3. Test it using the amazing papers supplied into folder *pdf*:

#### Add pdf files into database:
```
python document_storer.py
```

#### Query the newly-created database:
You can ask questions like:
- *What is Block Point domain-wall speed?*
- *What is a GROUP BY query?*
```
python retriever.py
```
