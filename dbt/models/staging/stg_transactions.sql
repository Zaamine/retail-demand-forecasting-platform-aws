select
    cast(date as date) as date,
    cast(store_nbr as bigint) as store_nbr,
    cast(transactions as bigint) as transactions
from {{ source('raw', 'transactions') }}
