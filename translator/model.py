from enum import Enum, unique

from pydantic import BaseModel, computed_field


@unique
class Languages(Enum):
    az = "az"
    bn = "bn"
    de = "de"
    em = "em"
    es = "es"
    fa = "fa"
    fr = "fr"
    he = "he"
    hu = "hu"
    id = "id"
    it = "it"
    ja = "ja"
    ko = "ko"
    pl = "pl"
    pt = "pt"
    ru = "ru"
    tr = "tr"
    uk = "uk"
    ur = "ur"
    vi = "vi"
    yo = "yo"
    zh = "zh"
    zh_hant = "zh-hant"


class TranslationDoc(BaseModel):
    official_lang: str = "en"
    translation_lang: str
    original_file: str
    translation_file: str | None = None
    translation_exists: bool


class Summary(BaseModel):
    lang: str
    files_analyzed: int = 0
    files_translated: int = 0
    files_outdated: int = 0
    files_missing_translation: int = 0
    files: list[TranslationDoc] = []

    @computed_field  # type: ignore
    @property
    def percentage_missing_translation(self) -> float:
        try:
            return (
                100 * float(self.files_missing_translation) / float(self.files_analyzed)
            )
        except Exception:
            return 0.0

    def append_file(self, doc: TranslationDoc) -> None:
        self.files.append(doc)
        self.files_analyzed += 1

        if doc.translation_exists:
            self.files_translated += 1

        if not doc.translation_exists:
            self.files_missing_translation += 1

    def first_missing_translation_files(self, length: int = 10) -> list[TranslationDoc]:
        return list(filter(lambda d: not d.translation_exists, self.files))[:length]
