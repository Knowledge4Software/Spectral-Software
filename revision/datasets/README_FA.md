# نسخهٔ صحیح دیتاست‌ها

برای ریویژن فقط بسته‌های زیر استفاده شوند. فایل‌های بزرگ بیرون مخزن Git نگه داشته
شده‌اند؛ برای مرتب‌سازی پروژه جابه‌جا، بازسازی یا دوباره فشرده نشده‌اند.

مسیر محلی: `C:\PyProjects\spectrals\data\benchmarks`

| دیتاست | ZIP مورد استفاده | اندازهٔ دقیق، بایت | train / valid / test |
| --- | --- | ---: | --- |
| CodeNet | `codenet_revision_common_support.zip` | 1473994602 | 69538 / 14896 / 14906 |
| AtCoder | `atcoder_revision_common_support_v2.zip` | 194769404 | 70000 / 15000 / 15000 |
| BigCloneBench | `bigclonebench_revision_common_support_v2.zip` | 45257983 | 70000 / 7129 / 7566 |

## آیا آپلود قبلی قابل استفاده است؟

**بله، اگر همین بستهٔ دقیق را قبلاً آپلود کرده‌ای.** Dataset جدید در Kaggle لازم
نیست؛ می‌توانی از همان نسخهٔ موجود استفاده کنی. اگر بستهٔ قدیمی بوده، نسخهٔ جدید
Dataset را با ZIP صحیح بساز و نوت‌بوک‌ها را به همان نسخه متصل کن.

نام نمایشی، slug و عبارت V3 در متادیتای منبع، یکسان بودن داده را ثابت نمی‌کند.
فایل‌های قدیمی `atcoder.zip` و `bigclonebench.zip` و پوشه‌های بدون پسوند `_v2`
برای مقایسهٔ جدید مناسب نیستند. از همهٔ روش‌های یک دیتاست باید دقیقاً روی یک support
استفاده شود. نسخهٔ CodeNet در ران موفق قبلی نیازی به تغییر ندارد.

AtCoder v2 زوج‌های بدون ترتیبِ تکراری را حذف کرده و sidecar نگاشت DDG به AST
را دارد. BigCloneBench v2 با حفظ labelها، اشتراک endpoint/source میان splitها
و زوج‌های تکراری را حذف می‌کند. این مجموعهٔ دوم تک‌زبانه است و problem ID معتبر
ندارد؛ جداسازی بر اساس مسئله یا تحلیل Type-IV برایش ادعا نمی‌شود.

## کنترل نسخه روی فایل‌های Kaggle

اگر ZIP اصلی را نگه داشته‌ای، SHA-256 آن باید مطابق [versions.json](versions.json)
و مقادیر زیر باشد:

```text
codenet_revision_common_support.zip
ec86d556b37b32180c44be50b3d79336fabae97432fe004a0baaebc7b9e4b46f

atcoder_revision_common_support_v2.zip
3525d3a750d395fd912da6879f96986dc4d9fa639fd78293c4c542c7d62d7318

bigclonebench_revision_common_support_v2.zip
b6bc759554b73379f61d1c2e3f0de63addfd28d68f2dae91a9d494c342ce9bd5
```

اگر Kaggle فایل‌ها را باز کرده است، SHA خود ZIP را نمی‌توان از پوشه مقایسه کرد.
`metadata.json` را بررسی کن: protocol، dataset key و hash جدول pair باید با
`versions.json` سازگار باشد؛ برای AtCoder، `ddg_ast_alignment` هم باید موجود باشد.
این کنترل اولیه جای بررسی تمام فایل‌ها را نمی‌گیرد. loader هنگام خواندن داده،
محتوای واقعی و alignment را نیز وارد بررسی و هویت آزمایش می‌کند.

فقط بستهٔ مربوط به همان دیتاست را Attach کن. `metadata.json`، `codes.jsonl.gz`،
`pairs.csv.gz`، `graph_spectra.jsonl.gz` و sidecarهای بسته باید قابل مشاهده باشند.
loader فایل‌های بازشدهٔ plain را هم می‌شناسد، ولی داخل ZIP بسته جست‌وجو نمی‌کند.

اسم پیشنهادی برای یافتن آسان در Kaggle:

- `SPECTRA CodeNet Revision Common Support v1`
- `SPECTRA AtCoder Revision Common Support v2`
- `SPECTRA BigCloneBench Revision Common Support v2`

شمارهٔ v2 داده مستقل از شمارهٔ v3 نوت‌بوک و سیاست بازیابی AMP است.
