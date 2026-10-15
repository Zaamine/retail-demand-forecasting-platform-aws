select
    cast(date as date) as date,
    type,
    locale,
    locale_name,
    description,
    cast(transferred as boolean) as transferred
from {{ source('raw', 'holidays_events') }}
