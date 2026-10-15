-- Multiple events may share a date, but fully duplicated event records are not expected.
select
    date,
    type,
    locale,
    locale_name,
    description,
    transferred,
    count(*) as row_count
from {{ ref('stg_holidays_events') }}
group by
    date,
    type,
    locale,
    locale_name,
    description,
    transferred
having count(*) > 1
