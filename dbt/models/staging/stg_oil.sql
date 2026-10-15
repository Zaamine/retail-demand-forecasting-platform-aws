select
    cast(date as date) as date,
    cast(dcoilwtico as double) as dcoilwtico
from {{ source('raw', 'oil') }}
