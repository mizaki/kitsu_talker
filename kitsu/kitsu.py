"""
Kitsu information source for comictagger
"""
# Copyright comictagger team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
from typing import Any, Callable, Generic, TypeVar
from urllib.parse import urljoin

import comictalker.talker_utils as talker_utils
import requests
import settngs
from comicapi import utils
from comicapi.genericmetadata import ComicSeries, GenericMetadata, MetadataOrigin
from comicapi.issuestring import IssueString
from comictalker.comiccacher import ComicCacher
from comictalker.comiccacher import Issue as CCIssue
from comictalker.comiccacher import Series as CCSeries
from comictalker.comictalker import ComicTalker, TalkerDataError, TalkerNetworkError
from pyrate_limiter import Duration, Limiter, RequestRate
from typing_extensions import Literal, TypedDict

logger = logging.getLogger(f"comictalker.{__name__}")

SEARCH_QUERY = """query mangaByTitle($title: String!, $limit: Int, $after: String) {
  searchMangaByTitle(first: $limit, title: $title, after: $after) {
    nodes {
      id
      startDate
      description
      sfw
      ageRating
      mangasub: subtype
      chapterCount
      chapterCountGuess
      volumeCount
      slug
      titles {
        canonical
        canonicalLocale
        localized
        alternatives
        original
        originalLocale
        translated
        romanized
      }
      productions(first: 5) {
        nodes {
          company {
            name
          }
        }
      }
      posterImage {
        original {
          url
        }
      }
      categories(first: 20) {
        nodes {
          title
          isNsfw
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}
"""

MANGA_ID_QUERY = """query mangaByID ($id: ID!) {
    findMangaById(id: $id) {
        id
        startDate
        endDate
        description
        status
        sfw
        mangasub: subtype
        ageRating
        endDate
        chapterCount
        chapterCountGuess
        volumeCount
        averageRating
        slug
        titles {
            canonical
            canonicalLocale
            localized
            alternatives
            original
            originalLocale
            translated
            romanized
        }
        posterImage {
            original {
                url
            }
        }
        categories(first: 20) {
          nodes {
            title
            isNsfw
          }
        }
        characters(first: 500) {
            nodes {
                role
                character {
                    id
                    names {
                        original
                        localized
                        canonical
                    }
                }
            }
        }
        originCountries
        originLanguages
        productions(first: 10) {
            nodes {
                company {
                    name
                }
                role
            }
        }
        staff(first: 50) {
            nodes {
                person {
                  name
                  names {
                    original
                    localized
                  }
                }
                role
            }
        }
    }
}
"""

CHAPTERS_QUERY = """query chaptersByMangaId ($id: ID!, $limit: Int, $after: String) {
    findMangaById(id: $id) {
        chapters (first: $limit, after: $after) {
            nodes {
                id
                titles {
                  original
                  localized
                  romanized
                  translated
                  alternatives
                  # canonical # Can be null which isn't allowed so errors
                }
                description
                number
                volume {
                    number
                }
                releasedAt
                length
                thumbnail {
                    original {
                        url
                    }
                }
            }
        totalCount
        pageInfo {
            hasNextPage
            endCursor
        }
        }
        chapterCount
        chapterCountGuess # Ongoing/guessed count
    }
}
"""

CHAPTER_QUERY = """query chapterByMangaId ($id: ID!, $number: Int!) {
    findMangaById(id: $id) {
        chapter (number: $number) {
            id
            titles {
              original
              localized
              romanized
              translated
              alternatives
              # canonical # Can be null which isn't allowed so errors
            }
            description
            number
            volume {
                number
            }
            releasedAt
            length
            thumbnail {
                original {
                    url
                }
            }
        }
        chapterCount
        chapterCountGuess # Ongoing/guessed count
    }
}"""

CHAPTER_QUERY_ID = """query chapterById ($id: ID!) {
    findChapterById(id: $id) {
        manga {id}
        id
        titles {
            original
            localized
            romanized
            translated
            alternatives
            # canonical # Can be null which isn't allowed so errors
        }
        description
        number
        volume {
            number
        }
        releasedAt
        length
        thumbnail {
            original {
                url
            }
        }
    }
}"""

TEST_GQL_QUERY = """query testGQL($id: ID!) {
  findMangaById(id: $id) {
    id
  }
}"""


class KitsuQLErrorLocations(TypedDict):
    line: int
    column: int


class KitsuQLErrorExtensions(TypedDict):
    code: str
    typeName: str
    fieldName: str


class KitsuGQLAPIErrors(TypedDict):
    message: str
    locations: list[KitsuQLErrorLocations]
    path: list[str]
    extensions: KitsuQLErrorExtensions


class KitsuError(TypedDict):
    errors: list[KitsuGQLAPIErrors]


class Character(TypedDict):
    id: str
    names: MangaTitles


class CharacterNode(TypedDict):
    role: str
    character: Character


class MangaCharacters(TypedDict):
    nodes: list[CharacterNode]


class ProductionNode(TypedDict):
    company: dict[Any, Any]
    role: str


class ProductionsData(TypedDict):
    nodes: list[ProductionNode]


class MangaProductions(TypedDict):
    productions: ProductionsData


class Person(TypedDict):
    name: str
    names: MangaTitles


class StaffNode(TypedDict):
    person: Person
    role: str


class MangaStaff(TypedDict):
    nodes: list[StaffNode]


class MangaTitles(TypedDict, total=False):
    canonical: str
    localized: dict[str, str]
    alternatives: list[str]
    original: str
    translated: str
    romanized: str


class MangaCategory(TypedDict):
    title: dict[str, str]
    isNsfw: bool


class MangaCategories(TypedDict):
    nodes: list[MangaCategory]


class MangaNode(TypedDict, total=False):
    id: str
    startDate: str
    endDate: str
    description: dict[str, str]
    status: Literal["current", "finished", "tba", "unreleased", "upcoming"]
    sfw: bool
    ageRating: Literal["G", "PG", "R", "R18"]
    mangasub: Literal["doujin", "manga", "manhua", "manhwa", "novel", "oel", "oneshot"]
    chapter: MangaChapter
    chapterCount: int | None
    chapterCountGuess: int | None
    volumeCount: int | None
    averageRating: float | None
    slug: str
    titles: MangaTitles
    posterImage: dict[str, dict[str, str]]
    categories: MangaCategories
    characters: MangaCharacters
    originCountries: str | None
    originLanguages: str | None
    productions: MangaProductions
    staff: MangaStaff


class MangaPageInfo(TypedDict):
    hasNextPage: bool
    endCursor: str


class MangaTotalCount(TypedDict):
    totalCount: int


class SearchMangaByTitleDict(TypedDict):
    nodes: list[MangaNode]
    pageInfo: MangaPageInfo
    totalCount: int


class MangaDataQL(TypedDict):
    searchMangaByTitle: SearchMangaByTitleDict


class MangaSingleDataQL(TypedDict):
    findMangaById: MangaNode


class MangaChapter(TypedDict, total=False):
    id: str
    manga: MangaNode
    titles: MangaTitles
    description: dict[str, str]
    number: int
    volume: dict[str, int | None]
    releasedAt: str | None
    length: int | None
    thumbnail: str | None


class MangaChaptersData(TypedDict):
    nodes: list[MangaChapter]
    totalCount: int
    pageInfo: MangaPageInfo


class FindMangaByIdDataChapters(TypedDict):
    chapters: MangaChaptersData
    chapterCount: int | None
    chapterCountGuess: int | None


class MangaDataChaptersQL(TypedDict):
    findMangaById: FindMangaByIdDataChapters


class MangaSingleChapterQL(TypedDict):
    findChapterById: MangaChapter


T = TypeVar("T", MangaDataQL, MangaSingleDataQL, MangaDataChaptersQL, MangaSingleChapterQL)


class ResponseQL(TypedDict, Generic[T]):
    data: T


# "As a general rule, no more than 3 requests per second. However, if you're making really expensive calls to the API,
# you may want to limit it more. The longer it takes to return, the more expensive it probably is." nucknyan via Discord
limiter = Limiter(RequestRate(3, Duration.SECOND))


class KitsuTalker(ComicTalker):
    name: str = "Kitsu"
    id: str = "kitsu"
    comictagger_min_ver: str = "1.6.0a7"
    website: str = "https://kitsu.io/"
    logo_url: str = "https://kitsu.io/kitsu-logo.png"
    attribution: str = f"Metadata provided by <a href='{website}'>{name}</a>"
    about: str = (
        f"Discover and share manga you love on <a href='{website}'>{name}</a>, the largest social anime "
        f"and manga tracker! Find new and interesting manga, discuss chapters you’ve watched or read, "
        f"and keep track of your progress and history."
        f"<p>NOTE: Issue number is used as chapter number!</p>"
    )
    # Used age ratings by kitsu
    age_range: list[str] = ["G", "PG", "R", "R18"]

    def __init__(self, version: str, cache_folder: pathlib.Path):
        super().__init__(version, cache_folder)
        # Default settings
        self.default_api_url = self.api_url = "https://kitsu.io/api/graphql"

        self.age_filter: str = "R"  # Same as Kitsu default
        self.age_filter_range: list[str] = []

        self.use_ongoing_issue_count: bool = False
        self.title_pref: str = "canonical"
        self.use_series_start_as_volume: bool = False
        self.use_series_desc: bool = False

    def register_settings(self, parser: settngs.Manager) -> None:
        parser.add_setting(
            f"--{self.id}-use-series-desc",
            default=False,
            action=argparse.BooleanOptionalAction,
            display_name="Use the series description",
            help="If the chapter description is empty, use the series description instead",
        )
        parser.add_setting(
            f"--{self.id}-use-ongoing",
            default=False,
            action=argparse.BooleanOptionalAction,
            display_name="Use the ongoing issue count",
            help='If a series is labelled as "ongoing", use the current issue count (otherwise empty)',
        )
        parser.add_setting(
            f"--{self.id}-use-series-start-as-volume",
            default=False,
            action=argparse.BooleanOptionalAction,
            display_name="Use series start as volume",
            help="Use the series start year as the volume number",
        )
        parser.add_setting(
            f"--{self.id}-age-filter",
            default="R",
            choices=["G", "PG", "R", "R18"],
            display_name="Age rating filter:",
            help="Select the level of age rating filtering. *Not guaranteed, relies on correct tagging*",
        )
        # "original" is based on the original_languages and original_countries fields, which are then looked up in the
        # "localized" titles hash (list of titles by locale code), "translated" attempts to find a title in your own
        # language. "preferred" is whatever the user preference is when logged in. "romanized" is en-t-ja
        parser.add_setting(
            f"--{self.id}-title-pref",
            default="canonical",
            choices=["canonical", "original", "localized", "romanized", "translated"],
            display_name="Naming preference:",
            help="Select the naming preference from; canonical, original, localized, romanized, translated. Will fallback to canonical",
        )
        parser.add_setting(f"--{self.id}-key", file=False, cmdline=False)
        parser.add_setting(
            f"--{self.id}-url",
            display_name="API URL",
            help=f"Use the given Kitsu GraphQL API URL. (default: {self.default_api_url})",
        )

    def parse_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        settings = super().parse_settings(settings)

        self.use_ongoing_issue_count = settings[f"{self.id}_use_ongoing"]
        self.use_series_start_as_volume = settings[f"{self.id}_use_series_start_as_volume"]
        self.use_series_desc = settings[f"{self.id}_use_series_desc"]
        self.title_pref = settings[f"{self.id}_title_pref"]
        self.age_filter = settings[f"{self.id}_age_filter"]

        # Create a filter with all accepted age rating
        self.age_filter_range = KitsuTalker.age_range[: KitsuTalker.age_range.index(self.age_filter) + 1]

        return settings

    def check_status(self, settings: dict[str, Any]) -> tuple[str, bool]:
        url = talker_utils.fix_url(settings[f"{self.id}_url"])
        if not url:
            url = self.default_api_url

        try:
            data = {"query": TEST_GQL_QUERY, "variables": {"id": 35}}
            # Further check the response?
            kitsu_response = self._get_content(url, data)
            return "API test successful", True
        except Exception:
            return f"GraphQL test failed for URL: {url}", False

    def search_for_series(
        self,
        series_name: str,
        callback: Callable[[int, int], None] | None = None,
        refresh_cache: bool = False,
        literal: bool = False,
        series_match_thresh: int = 90,
    ) -> list[ComicSeries]:
        search_series_name = utils.sanitize_title(series_name, literal)
        logger.info(f"{self.name} searching: {search_series_name}")

        # Before we search online, look in our cache, since we might have done this same search recently
        # For literal searches always retrieve from online
        cvc = ComicCacher(self.cache_folder, self.version)
        if not refresh_cache and not literal:
            cached_search_results = cvc.get_search_results(self.id, series_name)
            if len(cached_search_results) > 0:
                # Unpack to apply any filters
                json_cache: list[MangaNode] = [json.loads(x[0].data) for x in cached_search_results]
                json_cache = self._filter_series(json_cache)
                return self._format_search_results(json_cache)

        # "GraphQL limit is 2k in general (so long as it's below 500k nodes)" but search maxes at 20, bug?
        variables = {"title": search_series_name, "limit": 20, "after": ""}  # Use endCursor string for next page

        data = {
            "query": SEARCH_QUERY,
            "variables": variables,
        }

        kitsu_response: ResponseQL[MangaDataQL] = self._get_content(self.api_url, data)

        search_results: list[MangaNode] = []

        current_result_count = len(kitsu_response["data"]["searchMangaByTitle"]["nodes"])
        total_result_count: int = kitsu_response["data"]["searchMangaByTitle"]["totalCount"]  # Bugged, max always 20?

        # 1. Don't fetch more than some sane amount of pages.
        # 2. Halt when any result on the current page is less than or equal to a set ratio using thefuzz
        max_results: int = 100  # 5 pages

        total_result_count = min(total_result_count, max_results)

        if callback is None:
            logger.debug(f"Found {total_result_count} results")
        search_results.extend(kitsu_response["data"]["searchMangaByTitle"]["nodes"])

        if callback is not None:
            callback(current_result_count, total_result_count)

        # see if we need to keep asking for more pages...
        while current_result_count < total_result_count:
            if not literal:
                # Stop searching once any entry falls below the threshold
                stop_searching = any(
                    not utils.titles_match(search_series_name, series["titles"]["canonical"], series_match_thresh)
                    for series in kitsu_response["data"]["searchMangaByTitle"]["nodes"]
                )

                if stop_searching:
                    break

            if callback is None:
                logger.debug(f"getting another page of results 20 of {total_result_count}...")

            # Make sure there is another page
            if kitsu_response["data"]["searchMangaByTitle"]["pageInfo"]["hasNextPage"]:
                after = kitsu_response["data"]["searchMangaByTitle"]["pageInfo"]["endCursor"]
                variables = {
                    "title": search_series_name,
                    "limit": 20,
                    "after": after,  # Use endCursor string for next page
                }

                data = {
                    "query": SEARCH_QUERY,
                    "variables": variables,
                }

                kitsu_response = self._get_content(self.api_url, data)

                search_results.extend(kitsu_response["data"]["searchMangaByTitle"]["nodes"])
                current_result_count += len(kitsu_response["data"]["searchMangaByTitle"]["nodes"])

                if callback is not None:
                    callback(current_result_count, total_result_count)
            else:
                break

        # Cache raw data
        cvc.add_search_results(
            self.id,
            series_name,
            [CCSeries(id=x["id"], data=json.dumps(x).encode("utf-8")) for x in search_results],
            False,
        )

        # Apply any filters
        search_results = self._filter_series(search_results)

        # Format result to ComicSeries
        formatted_search_results = self._format_search_results(search_results)

        return formatted_search_results

    def fetch_comic_data(
        self, issue_id: str | None = None, series_id: str | None = None, issue_number: str = ""
    ) -> GenericMetadata:
        comic_data = GenericMetadata()
        if issue_id:
            comic_data = self._fetch_issue_data_by_issue_id(issue_id)
        elif issue_number and series_id:
            comic_data = self._fetch_issue_data(series_id, int(issue_number))

        return comic_data

    def fetch_issues_in_series(self, series_id: str) -> list[GenericMetadata]:
        cvc = ComicCacher(self.cache_folder, self.version)
        cached_series_issues_result = cvc.get_series_issues_info(series_id, self.id)

        series_data: MangaNode = self._fetch_series(series_id)

        if (
            len(cached_series_issues_result) == series_data["chapterCount"]
            or len(cached_series_issues_result) == series_data["chapterCountGuess"]
        ):
            return [
                self._map_comic_issue_to_metadata(json.loads(x[0].data), series_data)
                for x in cached_series_issues_result
            ]

        variables = {"id": series_id, "limit": 500, "after": ""}  # Use endCursor string for next page

        data = {
            "query": CHAPTERS_QUERY,
            "variables": variables,
        }

        kitsu_response: ResponseQL[MangaDataChaptersQL] = self._get_content(self.api_url, data)
        kitsu_issues: list[MangaChapter] = kitsu_response["data"]["findMangaById"]["chapters"]["nodes"]

        current_result_count: int = len(kitsu_issues)
        total_result_count: int = (
            kitsu_response["data"]["findMangaById"]["chapterCount"]
            or kitsu_response["data"]["findMangaById"]["chapterCountGuess"]
        ) or current_result_count

        # see if we need to keep asking for more pages...
        while current_result_count < total_result_count:
            # Make sure there is another page
            if kitsu_response["data"]["findMangaById"]["chapters"]["pageInfo"]["hasNextPage"]:
                after = kitsu_response["data"]["findMangaById"]["chapters"]["pageInfo"]["endCursor"]
                variables = {"id": series_id, "limit": 500, "after": after}  # Use endCursor string for next page

                data = {
                    "query": CHAPTERS_QUERY,
                    "variables": variables,
                }

                kitsu_response = self._get_content(self.api_url, data)

                kitsu_issues.extend(kitsu_response["data"]["findMangaById"]["chapters"]["nodes"])
                current_result_count += len(kitsu_response["data"]["findMangaById"]["chapters"]["nodes"])

        cvc.add_issues_info(
            self.id,
            [CCIssue(id=x["id"], series_id=series_id, data=json.dumps(x).encode("utf-8")) for x in kitsu_issues],
            True,
        )

        formatted_series_issues_result = [self._map_comic_issue_to_metadata(x, series_data) for x in kitsu_issues]

        return formatted_series_issues_result

    def fetch_issues_by_series_issue_num_and_year(
        self, series_id_list: list[str], issue_number: str, year: str | int | None
    ) -> list[GenericMetadata]:
        issues: list[GenericMetadata] = []

        for series_id in series_id_list:
            # TODO Add year once the filter is supported
            issues.append(self._fetch_issue_data(series_id, int(issue_number)))

        return issues

    @limiter.ratelimit("default", delay=True)
    def _get_content(self, url: str, params: dict[str, Any]) -> ResponseQL[T] | KitsuError:
        kitsu_response: ResponseQL[T] | KitsuError = self._get_url_content(url, params)
        if kitsu_response.get("errors"):
            logger.debug(f"{self.name} query failed with error: {json.dumps(kitsu_response, indent=2)}")
            raise TalkerNetworkError(self.name, 0, "Query error, check log for details")

        return kitsu_response

    def _get_url_content(self, url: str, params: dict[str, Any]) -> Any:
        for tries in range(3):
            try:
                # Uses "Accept-Language" header for localized titles etc. or the GQL request can use "locales: ["*"]"
                headers = {
                    "Accept-Language": "en",
                    "user-agent": "comictagger/" + self.version,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
                resp = requests.post(url, json=params, headers=headers)

                if resp.status_code == requests.status_codes.codes.ok:
                    return resp.json()
                if resp.status_code == requests.status_codes.codes.server_error:
                    logger.debug(f"Try #{tries + 1}: ")
                    time.sleep(1)
                    logger.debug(str(resp.status_code))
                if resp.status_code == requests.status_codes.codes.bad_request:
                    logger.debug(f"Bad request: {resp.json()}")
                    raise TalkerNetworkError(self.name, 2, "Bad request, see log for details")
                if resp.status_code == requests.status_codes.codes.forbidden:
                    logger.debug(f"Forbidden: {resp.json()}")
                    raise TalkerNetworkError(self.name, 2, "Forbidden, see log for details")
                if resp.status_code == requests.status_codes.codes.unauthorized:
                    logger.debug(f"Unauthorized: {resp.json()}")
                    raise TalkerNetworkError(self.name, 2, "Unauthorized, see log for details")
                if resp.status_code == requests.status_codes.codes.not_found:
                    logger.debug(f"Item not found: {resp.json()}")
                    raise TalkerNetworkError(self.name, 2, "Item not found, see log for details")

            except requests.exceptions.Timeout:
                logger.debug(f"Connection to {self.name} timed out.")
                raise TalkerNetworkError(self.name, 4)
            except requests.exceptions.RequestException as e:
                logger.debug(f"Request exception: {e}")
                raise TalkerNetworkError(self.name, 0, str(e)) from e
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error: {e}")
                raise TalkerDataError(self.name, 2, f"{self.name} did not provide json")

        raise TalkerNetworkError(self.name, 5)

    def _format_search_results(self, search_results: list[MangaNode]) -> list[ComicSeries]:
        formatted_results = []
        for record in search_results:
            title = self._prefered_title(record["titles"])

            alias_list = set(record["titles"]["alternatives"])
            # Add all localised titles, should include all others; original, translated, romanized, etc.
            for iso, alt_title in record["titles"]["localized"].items():
                if alt_title:
                    alias_list.add(alt_title)

            # productions/serializations currently empty
            pub_name = ""

            start_year = utils.xlate_int(record["startDate"][:4]) if record.get("startDate") else None

            formatted_results.append(
                ComicSeries(
                    aliases=alias_list,
                    count_of_issues=utils.xlate_int(record.get("chapterCount"))
                    or utils.xlate_int(record.get("chapterCountGuess")),
                    count_of_volumes=utils.xlate_int(record.get("volumeCount")),
                    description=record["description"].get("en", ""),
                    id=str(record["id"]),
                    image_url=record["posterImage"]["original"].get("url", ""),
                    name=title,
                    publisher=pub_name,
                    start_year=start_year,
                    format=utils.xlate(record.get("mangasub")),
                )
            )

        return formatted_results

    def _filter_series(self, series_results: list[MangaNode]) -> list[MangaNode]:
        def is_filtered(record: MangaNode) -> bool:
            # R is marked as sfw
            if "R18" not in self.age_filter_range and not record.get("sfw"):
                return True
            # Not every record has an age rating
            if record.get("ageRating") and record["ageRating"] not in self.age_filter_range:
                return True
            # It's possible that it's marked sfw and has a null age rating but has a isNsfw category
            if "R18" not in self.age_filter_range:
                for cat in record["categories"].get("nodes", []):
                    if cat.get("isNsfw"):
                        return True
            return False

        return [series for series in series_results if not is_filtered(series)]

    def _prefered_title(self, titles: MangaTitles, use_localized: bool = False) -> str | None:
        title = None
        # BUG: canonical can be null which isn't allowed in GQL and the query will fail
        # https://github.com/hummingbird-me/kitsu-server/issues/1265
        # Work around bugged canonical, use localized in lieu
        if self.title_pref == "localized" or use_localized:
            # Take first item value as there could be en, en-us, ja-t-en, etc.
            if localized_titles := titles.get("localized"):
                title = next((title for title in localized_titles.values() if title), title)
        elif self.title_pref == "original":
            title = titles.get("original")
        elif self.title_pref == "romanized":
            title = titles.get("romanized")
        elif self.title_pref == "translated":
            title = titles.get("translated")

        # Fallback to canonical (if possible)
        if not title and titles.get("canonical"):
            title = titles["canonical"]

        return title

    def fetch_series(self, series_id: str) -> ComicSeries:
        return self._format_search_results([self._fetch_series(series_id)])[0]

    def _fetch_series(self, series_id: str) -> MangaNode:
        # Will fetch characters, staff etc.
        cvc = ComicCacher(self.cache_folder, self.version)
        cached_series_result = cvc.get_series_info(series_id, self.id)
        if cached_series_result is not None and cached_series_result[1]:
            return json.loads(cached_series_result[0].data)

        variables = {
            "id": series_id,
        }

        data = {
            "query": MANGA_ID_QUERY,
            "variables": variables,
        }

        kitsu_response: ResponseQL[MangaSingleDataQL] = self._get_content(self.api_url, data)

        if kitsu_response["data"]["findMangaById"]:
            cvc.add_series_info(
                self.id,
                CCSeries(id=series_id, data=json.dumps(kitsu_response["data"]["findMangaById"]).encode("utf-8")),
                True,
            )

        return kitsu_response["data"]["findMangaById"]

    def _fetch_issue_data(self, series_id: str, issue_number: int) -> GenericMetadata:
        # issue number presumed to be chapter number
        # Can't fetch from cache with series id and issue number
        cvc = ComicCacher(self.cache_folder, self.version)

        variables = {
            "id": series_id,
            "number": issue_number,
        }

        data = {
            "query": CHAPTER_QUERY,
            "variables": variables,
        }

        kitsu_response: ResponseQL[MangaSingleDataQL] = self._get_content(self.api_url, data)

        # It is possible this can be null
        if kitsu_response["data"]["findMangaById"]["chapter"] is None:
            return GenericMetadata()

        if kitsu_response:
            cvc.add_issues_info(
                self.id,
                [
                    CCIssue(
                        id=kitsu_response["data"]["findMangaById"]["chapter"]["id"],
                        series_id=str(series_id),
                        data=json.dumps(kitsu_response["data"]["findMangaById"]["chapter"]).encode("utf-8"),
                    )
                ],
                True,
            )

        if kitsu_response["data"]["findMangaById"]["chapter"]:
            return self._map_comic_issue_to_metadata(
                kitsu_response["data"]["findMangaById"]["chapter"], self._fetch_series(series_id)
            )

        return GenericMetadata()

    def _fetch_issue_data_by_issue_id(self, issue_id: str) -> GenericMetadata:
        cvc = ComicCacher(self.cache_folder, self.version)
        cached_issues_result = cvc.get_issue_info(issue_id, self.id)

        if cached_issues_result and cached_issues_result[1]:
            return self._map_comic_issue_to_metadata(
                json.loads(cached_issues_result[0].data), self._fetch_series(cached_issues_result[0].series_id)
            )

        variables = {
            "id": issue_id,
        }

        data = {
            "query": CHAPTER_QUERY_ID,
            "variables": variables,
        }

        kitsu_response: ResponseQL[MangaSingleChapterQL] = self._get_content(self.api_url, data)

        # It is possible this can be null
        if kitsu_response["data"]["findChapterById"] is None:
            return GenericMetadata()

        if kitsu_response:
            cvc.add_issues_info(
                self.id,
                [
                    CCIssue(
                        id=issue_id,
                        series_id=str(kitsu_response["data"]["findChapterById"]["manga"]["id"]),
                        data=json.dumps(kitsu_response["data"]["findChapterById"]).encode("utf-8"),
                    )
                ],
                True,
            )

        if kitsu_response["data"]["findChapterById"]:
            return self._map_comic_issue_to_metadata(
                kitsu_response["data"]["findChapterById"],
                self._fetch_series(str(kitsu_response["data"]["findChapterById"]["manga"]["id"])),
            )

        return GenericMetadata()

    def _map_comic_issue_to_metadata(self, issue: MangaChapter, series: MangaNode) -> GenericMetadata:
        md = GenericMetadata(
            data_origin=MetadataOrigin(self.id, self.name),
            issue_id=utils.xlate(issue["id"]),
            series_id=utils.xlate(series["id"]),
            issue=utils.xlate(IssueString(str(issue["number"])).as_string()),
        )
        md.series = self._prefered_title(series["titles"])

        md.manga = "Yes"

        # Check if series is ongoing to legitimise issue count OR use option setting
        if series["status"] == "FINISHED" or self.use_ongoing_issue_count:
            md.issue_count = utils.xlate_int(series["chapterCount"]) or utils.xlate_int(series["chapterCountGuess"])
            md.volume_count = utils.xlate_int(series["volumeCount"])

        # Will take first item value (possibly use language selection later)
        # By default a 'en': '' is added
        md.description = next(iter(issue["description"].values()), "")
        if self.use_series_desc and not md.description:
            md.description = next(iter(series["description"].values()))

        for cat in series["categories"].get("nodes", []):
            # Can be null
            if cat:
                md.genres.add(next(iter(cat["title"].values())))

        # Want to keep anything but manga?
        md.format = (
            series["mangasub"].capitalize() if series.get("mangasub") and series["mangasub"] != "MANGA" else None
        )

        md.title = self._prefered_title(issue["titles"], True)

        md.series_aliases = set(series["titles"].get("alternatives", set()))
        # Add all localised titles, should already include all others; original, translated, romanized, etc.
        for iso, title in series["titles"]["localized"].items():
            if title:
                md.series_aliases.add(title)

        md.maturity_rating = series.get("ageRating")

        # Have to build the link
        md.web_link = urljoin(self.website, f"manga/{series['slug']}/chapters/{issue['number']}")

        # Convert from 0-100 to 0-5
        md.critical_rating = series["averageRating"] / 20.0 if series["averageRating"] is not None else None

        # Support localized? All other "names" are null/invalid
        for credit in series["staff"]["nodes"]:
            roles = credit["role"].split("&")
            for role in roles:
                if role.strip() == "Art":
                    role = "Artist"
                if role.strip() == "Story":
                    role = "Writer"
                md.add_credit(credit["person"]["name"], role, True)

        for character in series["characters"]["nodes"]:
            md.characters.add(self._prefered_title(character["character"]["names"]))

        md.volume = utils.xlate_int(issue["volume"].get("number")) if issue.get("volume") else None

        if self.use_series_start_as_volume:
            md.volume = utils.xlate_int(series["startDate"][:4]) if series.get("startDate") else None

        if issue.get("releasedAt"):
            # ISO 8601 date YYYY-MM-DD
            md.day, md.month, md.year = utils.parse_date_str(issue["releasedAt"])
        elif series.get("startDate"):
            md.year = utils.xlate_int(series["startDate"][:4])

        return md
