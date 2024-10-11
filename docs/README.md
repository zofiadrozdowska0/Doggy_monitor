**README - Jak uruchomić MkDocs na swojej maszynie**

Poniższa instrukcja pokaże, jak skonfigurować i uruchomić MkDocs – narzędzie do generowania statycznej dokumentacji z plików w formacie Markdown.

**1. Instalacja MkDocs**

Aby korzystać z MkDocs, potrzebujesz mieć zainstalowanego **Pythona** i **pip**. Jeśli nie masz ich na swojej maszynie, możesz je pobrać z oficjalnej strony [Python](https://www.python.org/).

**Instalacja MkDocs:**

Użyj pip, aby zainstalować MkDocs:

`pip install mkdocs`

**3. Uruchomienie lokalnego serwera dokumentacji**

Aby przeglądać dokumentację na swojej maszynie, możesz uruchomić lokalny serwer za pomocą:

```sh
cd docs
mkdocs serve
```

Serwer uruchomi się na domyślnym porcie (8000). Możesz go otworzyć w przeglądarce pod adresem: [http://127.0.0.1:8000](http://127.0.0.1:8000).

**4. Edycja dokumentacji**

Plik index.md w folderze docs to domyślny plik startowy Twojej dokumentacji. Możesz go edytować przy użyciu edytora tekstowego, takiego jak VSCode, Notepad++, lub dowolnego innego.

Możesz dodawać nowe pliki Markdown w folderze docs i edytować plik mkdocs.yml, aby skonfigurować układ i nawigację dokumentacji.

**5. Budowanie dokumentacji**

Jeśli chcesz wygenerować statyczne pliki HTML dla swojej dokumentacji, użyj:

```shell
mkdocs build
```
